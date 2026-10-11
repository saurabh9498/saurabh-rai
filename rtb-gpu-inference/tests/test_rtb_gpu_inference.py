"""Checks that every number the README quotes is what the code computes."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
for sub in ("model", "sizing", "sim"):
    sys.path.insert(0, str(ROOT / sub))

import hot_cache_sim as hc  # noqa: E402
import latency_budget as lb  # noqa: E402
import training_memory as tm  # noqa: E402
import vram_budget as vb  # noqa: E402
from dlrm import DLRM, DLRMConfig, synthetic_batch  # noqa: E402


def test_split_model_equals_full_model():
    model = DLRM(DLRMConfig(vocab_size=1000)).eval()
    dense, sparse = synthetic_batch(model.cfg, 16)
    with torch.no_grad():
        full = model(dense, sparse)
        split = model.scorer(dense, model.embeddings(sparse))
    torch.testing.assert_close(full, split)
    assert full.shape == (16, 1) and torch.all((full > 0) & (full < 1))


def test_onnx_export_matches_torch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from export_onnx import export

    cfg = DLRMConfig(vocab_size=1000)
    path = export(tmp_path / "m.onnx", cfg, seed=3)
    torch.manual_seed(3)
    model = DLRM(cfg).eval()
    dense, sparse = synthetic_batch(cfg, 5, seed=3)
    with torch.no_grad():
        emb = model.embeddings(sparse)
        ref = model.scorer(dense, emb).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"dense": dense.numpy(), "embeddings": emb.numpy()})[0]
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_vram_budget_numbers():
    s = vb.summarize(vb.Budget())
    assert s["embeddings_int8_gb"] == pytest.approx(300)
    assert s["cold_tail_in_redis_gb"] == pytest.approx(293)
    assert s["hot_fraction_of_rows"] == pytest.approx(7 / 300)
    with pytest.raises(ValueError):
        vb.summarize(vb.Budget(hot_cache_gb=8))      # 5 + 12 + 8 > 24


def test_latency_budget_and_p99_logic():
    s = lb.summarize(0.98)
    assert s["hot_path_p99_ms"] == 45 and s["cold_path_p99_ms"] == 60
    assert s["cold_request_fraction"] == pytest.approx(1 - 0.98 ** 2)   # per-lookup -> per-request
    assert s["blended_p99_upper_bound_ms"] == 60                        # ~4% cold > 1% tail
    assert lb.summarize(0.996)["blended_p99_upper_bound_ms"] == 45
    assert lb.hit_rate_for_hot_p99(2) == pytest.approx(0.99499, abs=1e-5)
    assert lb.hit_rate_for_hot_p99(26) == pytest.approx(0.99961, abs=1e-5)


def test_training_memory_numbers():
    s = tm.summarize()
    assert s["naive_adam_training_tb"] == pytest.approx(4.8)
    assert s["node_hbm_gb"] == 640
    assert s["hot_rows_gb"] == pytest.approx(36)
    assert s["hot_row_replication_hbm_cost_gb"] == pytest.approx(288)
    assert s["checkpoint_bounce_buffer_s"] == pytest.approx(240)
    assert s["checkpoint_gds_s"] == pytest.approx(26.7, abs=0.1)


@pytest.mark.parametrize("s", [0.9, 1.1])
def test_hot_cache_analytic_matches_monte_carlo(s):
    a = hc.analytic_hit_rate(1_000_000, 0.0234, s)
    m = hc.monte_carlo_hit_rate(1_000_000, 0.0234, s, lookups=2_000_000)
    assert abs(a - m) < 0.002


def test_skew_for_98pct_and_training_skew():
    assert 1.15 < hc.skew_for_target(2.34e9, 7 / 300, 0.98) < 1.2
    assert hc.skew_for_target(2.34e9, 0.03, 0.80) == pytest.approx(0.973, abs=0.002)


@pytest.mark.parametrize("name", ["triton-deployment.yaml", "keda-scaledobject.yaml"])
def test_k8s_manifests_parse(name):
    doc = yaml.safe_load((ROOT / "deploy" / "k8s" / name).read_text())
    assert doc["kind"] in {"Deployment", "ScaledObject"}
    if doc["kind"] == "ScaledObject":
        assert doc["spec"]["triggers"][0]["metricType"] == "Value"


def test_startup_probe_covers_five_minutes():
    doc = yaml.safe_load((ROOT / "deploy" / "k8s" / "triton-deployment.yaml").read_text())
    c = doc["spec"]["template"]["spec"]["containers"][0]
    probe = c["startupProbe"]
    assert probe["periodSeconds"] * probe["failureThreshold"] == 300
    assert c["lifecycle"]["preStop"]["exec"]["command"][-1] == "sleep 5"
