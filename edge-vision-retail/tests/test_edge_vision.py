"""Quantization, per-store calibration, offline buffer, staged rollout and sizing."""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.buffer import Event, StoreAndForward, hours_of_outage  # noqa: E402
from fleet.rollout import WAVES, Device, Health, Rollout, wave_of  # noqa: E402
from model.cnn import accuracy, train  # noqa: E402
from quantize import ptq  # noqa: E402
from sizing import edge_sizing  # noqa: E402
from stores import adapt  # noqa: E402


@pytest.fixture(scope="session")
def trained():
    return train(seed=0, epochs=25)


# --- quantization ----------------------------------------------------------------------------
def test_bn_folding_is_exact(trained):
    model, (_, _, xte, _) = trained
    with torch.no_grad():
        torch.testing.assert_close(ptq.fold_bn(model)(xte), model(xte), rtol=1e-4, atol=1e-4)


def test_per_channel_error_never_exceeds_per_tensor():
    w = torch.randn(32, 9) * torch.logspace(0, 2, 32)[:, None]
    for bits in (8, 4):
        e_t = ((ptq.quantize_weight(w, bits, False) - w) ** 2).sum()
        e_c = ((ptq.quantize_weight(w, bits, True) - w) ** 2).sum()
        assert e_c <= e_t


def test_range_sweep_shape():
    rows = ptq.range_sweep()
    per_t = [float(r["mean_channel_rel_error_per_tensor"]) for r in rows]
    per_c = [float(r["mean_channel_rel_error_per_channel"]) for r in rows]
    assert per_t == sorted(per_t) and per_t[-1] > 1000 * per_t[0] / 2      # grows with the spread
    assert max(per_c) < 2 * min(per_c)                                       # stays flat
    assert rows[-1]["weights_rounded_to_zero_per_tensor"] > 0.2


def test_quantized_accuracy(trained):
    out = ptq.run(trained=trained)["accuracy"]
    assert out["fp32_bn_folded"] == out["fp32"]
    assert out["w8_per_channel_a8"] >= out["w8_per_tensor_a8"]
    assert out["w4_per_channel_a8"] > out["w4_per_tensor_a8"]
    assert out["fp32"] - out["w8_per_channel_a8"] <= 0.005


def test_activation_quantizer_keeps_negative_inputs():
    q = ptq.ActQuant(8)
    x = torch.linspace(-0.5, 1.5, 101)
    q.calibrating = True
    q(x)
    q.calibrating = False
    assert (q(x) - x).abs().max() < 2.0 / 255


# --- per-store calibration ----------------------------------------------------------------------
def test_calibration_uses_no_labels_and_touches_only_bn_stats(trained):
    model, (xtr, _, _, _) = trained
    cam, _ = adapt.store_camera(3)
    cal = adapt.calibrate_bn(model, cam(xtr[:200]))
    for (n, p), (_, p0) in zip(cal.named_parameters(), model.named_parameters()):
        assert torch.equal(p, p0), n                        # weights unchanged
    changed = [n for n, b in cal.named_buffers() if "running" in n and not torch.equal(b, dict(model.named_buffers())[n])]
    assert changed                                          # only the BN statistics moved


def test_one_model_plus_calibration_recovers_store_accuracy(trained):
    out = adapt.run(stores=6, store_specific=False, trained=trained)
    m = out["mean_accuracy"]
    assert m["calibrated"] > m["generic"] + 0.2
    assert abs(m["calibrated_int8"] - m["calibrated"]) < 0.01
    sizes = out["per_store_artifact_values"]
    assert sizes["store_specific_model"] / sizes["calibration"] > 100


# --- offline buffer -------------------------------------------------------------------------------
def ev(i, kind="count", size=100, ts=None):
    return Event(f"e{i}", float(i if ts is None else ts), kind, b"x" * size)


def test_buffer_forwards_in_order_and_dedupes():
    b = StoreAndForward()
    for i in (3, 1, 2):
        assert b.put(ev(i))
    assert not b.put(ev(2))                                 # duplicate ID ignored
    got = []
    assert b.drain(lambda batch: got.extend(e.event_id for e in batch) or True, batch=2) == 3
    assert got == ["e1", "e2", "e3"] and len(b) == 0


def test_failed_send_keeps_everything():
    b = StoreAndForward()
    for i in range(5):
        b.put(ev(i))
    assert b.drain(lambda batch: False) == 0 and len(b) == 5


def test_full_buffer_drops_least_important_oldest_first():
    b = StoreAndForward(capacity_bytes=3 * 164)             # room for three 100-byte events
    b.put(ev(1, "frame_meta"))
    b.put(ev(2, "alert"))
    b.put(ev(3, "count"))
    assert b.put(ev(4, "count"))                            # evicts the frame_meta
    assert b.dropped["frame_meta"] == 1
    assert b.put(ev(5, "alert"))                            # evicts the oldest count
    assert not b.put(ev(6, "frame_meta"))                   # never evicts something more important
    kinds = sorted(r[0] for r in b.db.execute("SELECT kind FROM events"))
    assert kinds == ["alert", "alert", "count"]


def test_refused_event_does_not_destroy_anything():
    b = StoreAndForward(capacity_bytes=2 * 164 + 50)
    b.put(ev(1, "frame_meta"))
    b.put(ev(2, "alert"))
    before = len(b)
    assert not b.put(ev(3, "frame_meta", size=400))          # cannot fit without evicting the alert
    assert len(b) == before and b.dropped == {"frame_meta": 1, "count": 0, "alert": 0}


def test_buffer_survives_restart(tmp_path):
    path = str(tmp_path / "buf.db")
    b = StoreAndForward(path)
    b.put(ev(1))
    b.db.close()
    assert len(StoreAndForward(path)) == 1


def test_outage_hours():
    rates = [("count", 1000, 436)]                          # 500 bytes/event incl. overhead
    assert hours_of_outage(1000 * 500 * 72, rates) == pytest.approx(72)


# --- staged rollout ---------------------------------------------------------------------------------
def fleet(n=300):
    return [Device(f"store-{i:03d}") for i in range(n)]


def healthy_probe(d, version):
    return Health(True, 30.0, 100.0)


def test_good_release_completes_every_wave():
    devs = fleet()
    r = Rollout(devs, healthy_probe)
    assert r.run("v2") == "completed"
    assert all(d.active == "v2" and d.standby == "v1" for d in devs)
    assert len([m for m in r.log if m.startswith("wave")]) == len(WAVES)


def _bad_on(bad_hw):
    def probe(d, version):
        slow = version == "v2" and d.device_id in bad_hw
        return Health(True, 18.0 if slow else 30.0, 100.0)
    return probe


def test_regression_on_canary_hardware_is_caught_in_the_canary_wave():
    devs = fleet()
    r = Rollout(devs, _bad_on({d.device_id for d in devs[::7]}))
    assert r.run("v2") == "rolled_back"
    assert all(d.active == "v1" for d in devs)
    assert "in the 1% wave" in r.log[0] and "fps dropped 40%" in r.log[0]


def test_regression_missing_from_canary_is_caught_later_and_everything_rolls_back():
    devs = fleet()
    canary = {d.device_id for d in devs if wave_of(d.device_id) < WAVES[0]}
    later = [d for d in devs if WAVES[0] <= wave_of(d.device_id) < WAVES[1]]
    bad = {later[-1].device_id}                              # only one device in the 10% wave regresses
    assert not bad & canary
    r = Rollout(devs, _bad_on(bad))
    assert r.run("v2") == "rolled_back"
    assert "in the 10% wave" in r.log[-2]
    assert all(d.active == "v1" for d in devs)               # canary devices rolled back too
    assert int(r.log[-1].split()[2]) > len(canary)


def test_offline_device_never_gets_a_rolled_back_release():
    devs = fleet()
    devs[5].online = False

    def probe(d, version):
        return Health(version != "v2", 30.0, 100.0)

    r = Rollout(devs, probe)
    r.run("v2")
    r.on_reconnect(devs[5])
    assert devs[5].active == "v1"


def test_offline_device_catches_up_to_a_completed_release():
    devs = fleet()
    devs[5].online = False
    r = Rollout(devs, healthy_probe)
    r.run("v2")
    assert devs[5].active == "v1"
    r.on_reconnect(devs[5])
    assert devs[5].active == "v2"


def test_wave_assignment_is_stable():
    assert wave_of("store-042") == wave_of("store-042")


# --- sizing -------------------------------------------------------------------------------------------
def test_packing_and_outage_numbers():
    p = edge_sizing.packing()
    assert p["fits_with_headroom"] and p["compute_utilisation"] > p["memory_utilisation"]
    assert edge_sizing.max_cameras() == 22
    o = edge_sizing.outage_budget()
    assert o["disk_needed_mb"] == 570
