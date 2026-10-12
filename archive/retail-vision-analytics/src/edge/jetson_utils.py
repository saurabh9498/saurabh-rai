"""
NVIDIA Jetson Utilities for Retail Vision Analytics.

This module provides Jetson-specific utilities for:
- Device detection and capability reporting
- Power mode management
- Thermal monitoring and throttling
- Memory optimization
- DLA (Deep Learning Accelerator) configuration
- Performance profiling

Supported Devices:
- Jetson Orin NX/Nano
- Jetson AGX Orin
- Jetson Xavier NX/AGX
- Jetson Nano

Requires: JetPack 5.1+
"""

import os
import re
import time
import logging
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class JetsonModel(Enum):
    """Supported Jetson models."""
    ORIN_NX_16GB = "orin_nx_16gb"
    ORIN_NX_8GB = "orin_nx_8gb"
    ORIN_NANO_8GB = "orin_nano_8gb"
    ORIN_NANO_4GB = "orin_nano_4gb"
    AGX_ORIN_64GB = "agx_orin_64gb"
    AGX_ORIN_32GB = "agx_orin_32gb"
    XAVIER_NX = "xavier_nx"
    AGX_XAVIER = "agx_xavier"
    NANO = "nano"
    UNKNOWN = "unknown"


class PowerMode(Enum):
    """Jetson power modes."""
    MAXN = "MAXN"  # Maximum performance
    MODE_15W = "15W"
    MODE_10W = "10W"
    MODE_7W = "7W"
    MODE_5W = "5W"
    MODE_2W = "2W"  # Orin Nano only


@dataclass
class JetsonCapabilities:
    """Jetson device capabilities."""
    
    model: JetsonModel
    jetpack_version: str
    cuda_version: str
    tensorrt_version: str
    
    # Compute
    gpu_cores: int
    dla_cores: int
    cpu_cores: int
    cpu_freq_mhz: int
    
    # Memory
    ram_gb: float
    swap_gb: float
    
    # Power
    max_power_watts: float
    current_power_mode: PowerMode
    available_power_modes: List[PowerMode]
    
    # Encoders/Decoders
    nvdec_instances: int
    nvenc_instances: int
    max_decode_streams: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model.value,
            "jetpack_version": self.jetpack_version,
            "cuda_version": self.cuda_version,
            "tensorrt_version": self.tensorrt_version,
            "gpu_cores": self.gpu_cores,
            "dla_cores": self.dla_cores,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "max_power_watts": self.max_power_watts,
            "current_power_mode": self.current_power_mode.value,
            "max_decode_streams": self.max_decode_streams,
        }


@dataclass
class ThermalStatus:
    """Thermal status of Jetson device."""
    
    timestamp: float
    
    # Temperatures in Celsius
    cpu_temp: float
    gpu_temp: float
    soc_temp: float
    
    # Fan
    fan_speed_percent: float
    fan_pwm: int
    
    # Throttling status
    is_throttled: bool
    throttle_reason: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_temp": self.cpu_temp,
            "gpu_temp": self.gpu_temp,
            "soc_temp": self.soc_temp,
            "fan_speed_percent": self.fan_speed_percent,
            "is_throttled": self.is_throttled,
            "throttle_reason": self.throttle_reason,
        }


@dataclass
class PowerStatus:
    """Power consumption status."""
    
    timestamp: float
    
    # Power in milliwatts
    total_power_mw: float
    gpu_power_mw: float
    cpu_power_mw: float
    soc_power_mw: float
    
    # Voltage and current
    voltage_mv: float
    current_ma: float
    
    # Mode
    power_mode: PowerMode
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_power_mw": self.total_power_mw,
            "total_power_w": self.total_power_mw / 1000,
            "gpu_power_mw": self.gpu_power_mw,
            "cpu_power_mw": self.cpu_power_mw,
            "power_mode": self.power_mode.value,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics from Jetson device."""
    
    timestamp: float
    
    # GPU utilization
    gpu_util_percent: float
    gpu_freq_mhz: int
    
    # CPU utilization
    cpu_util_percent: float
    cpu_freq_mhz: int
    
    # Memory
    ram_used_mb: float
    ram_total_mb: float
    swap_used_mb: float
    
    # EMC (External Memory Controller)
    emc_util_percent: float
    emc_freq_mhz: int
    
    @property
    def ram_util_percent(self) -> float:
        """Calculate RAM utilization percentage."""
        if self.ram_total_mb == 0:
            return 0
        return (self.ram_used_mb / self.ram_total_mb) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "gpu_util_percent": self.gpu_util_percent,
            "gpu_freq_mhz": self.gpu_freq_mhz,
            "cpu_util_percent": self.cpu_util_percent,
            "ram_util_percent": self.ram_util_percent,
            "ram_used_mb": self.ram_used_mb,
            "emc_util_percent": self.emc_util_percent,
        }


class JetsonDevice:
    """
    Interface for NVIDIA Jetson device management.
    
    Provides methods for:
    - Device detection and capability reporting
    - Power mode management
    - Thermal monitoring
    - Performance metrics
    - DLA configuration
    
    Example:
        >>> device = JetsonDevice()
        >>> caps = device.get_capabilities()
        >>> print(f"Model: {caps.model.value}")
        >>> print(f"GPU Cores: {caps.gpu_cores}")
        >>> 
        >>> # Set power mode
        >>> device.set_power_mode(PowerMode.MAXN)
        >>> 
        >>> # Monitor thermals
        >>> thermal = device.get_thermal_status()
        >>> print(f"GPU Temp: {thermal.gpu_temp}°C")
    """
    
    # Jetson model specifications
    MODEL_SPECS = {
        JetsonModel.ORIN_NX_16GB: {
            "gpu_cores": 1024,
            "dla_cores": 2,
            "cpu_cores": 8,
            "max_power": 25,
            "max_decode_streams": 32,
        },
        JetsonModel.ORIN_NX_8GB: {
            "gpu_cores": 1024,
            "dla_cores": 2,
            "cpu_cores": 6,
            "max_power": 25,
            "max_decode_streams": 32,
        },
        JetsonModel.AGX_ORIN_64GB: {
            "gpu_cores": 2048,
            "dla_cores": 2,
            "cpu_cores": 12,
            "max_power": 60,
            "max_decode_streams": 64,
        },
        JetsonModel.AGX_ORIN_32GB: {
            "gpu_cores": 2048,
            "dla_cores": 2,
            "cpu_cores": 12,
            "max_power": 40,
            "max_decode_streams": 64,
        },
        JetsonModel.XAVIER_NX: {
            "gpu_cores": 384,
            "dla_cores": 2,
            "cpu_cores": 6,
            "max_power": 15,
            "max_decode_streams": 16,
        },
        JetsonModel.NANO: {
            "gpu_cores": 128,
            "dla_cores": 0,
            "cpu_cores": 4,
            "max_power": 10,
            "max_decode_streams": 8,
        },
    }
    
    # Path mappings for sysfs access
    SYSFS_PATHS = {
        "thermal": "/sys/devices/virtual/thermal",
        "power": "/sys/bus/i2c/drivers/ina3221",
        "gpu_freq": "/sys/devices/gpu.0/devfreq/17000000.gv11b",
        "cpu_freq": "/sys/devices/system/cpu/cpu0/cpufreq",
        "emc": "/sys/kernel/actmon_avg_activity/mc_all",
        "fan": "/sys/devices/pwm-fan",
    }
    
    def __init__(self):
        """Initialize Jetson device interface."""
        self._model: Optional[JetsonModel] = None
        self._capabilities: Optional[JetsonCapabilities] = None
        self._is_jetson = self._detect_jetson()
        
        if self._is_jetson:
            self._model = self._detect_model()
            logger.info(f"Detected Jetson: {self._model.value}")
        else:
            logger.warning("Not running on Jetson device")
    
    def _detect_jetson(self) -> bool:
        """Detect if running on a Jetson device."""
        # Check for Tegra in device tree
        dt_model = Path("/proc/device-tree/model")
        if dt_model.exists():
            try:
                model_str = dt_model.read_text().lower()
                return "nvidia" in model_str and ("jetson" in model_str or "tegra" in model_str)
            except Exception:
                pass
        
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True, text=True, timeout=5
            )
            return "tegra" in result.stdout.lower() or "orin" in result.stdout.lower()
        except Exception:
            pass
        
        # Check for JetPack version file
        if Path("/etc/nv_tegra_release").exists():
            return True
        
        return False
    
    def _detect_model(self) -> JetsonModel:
        """Detect specific Jetson model."""
        # Try device tree
        dt_model = Path("/proc/device-tree/model")
        if dt_model.exists():
            try:
                model_str = dt_model.read_text().lower()
                
                if "orin nx" in model_str:
                    # Check RAM to distinguish variants
                    ram = self._get_total_ram_gb()
                    if ram >= 15:
                        return JetsonModel.ORIN_NX_16GB
                    else:
                        return JetsonModel.ORIN_NX_8GB
                elif "orin nano" in model_str:
                    ram = self._get_total_ram_gb()
                    if ram >= 7:
                        return JetsonModel.ORIN_NANO_8GB
                    else:
                        return JetsonModel.ORIN_NANO_4GB
                elif "agx orin" in model_str:
                    ram = self._get_total_ram_gb()
                    if ram >= 60:
                        return JetsonModel.AGX_ORIN_64GB
                    else:
                        return JetsonModel.AGX_ORIN_32GB
                elif "xavier nx" in model_str:
                    return JetsonModel.XAVIER_NX
                elif "agx xavier" in model_str:
                    return JetsonModel.AGX_XAVIER
                elif "nano" in model_str:
                    return JetsonModel.NANO
            except Exception:
                pass
        
        return JetsonModel.UNKNOWN
    
    def _get_total_ram_gb(self) -> float:
        """Get total RAM in GB."""
        try:
            with open("/proc/meminfo", 'r') as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        except Exception:
            pass
        return 0
    
    def _run_command(self, cmd: List[str], timeout: int = 10) -> Optional[str]:
        """Run a command and return output."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Command failed: {cmd}, error: {e}")
        return None
    
    def _read_sysfs(self, path: str) -> Optional[str]:
        """Read a sysfs file."""
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except Exception:
            return None
    
    def _write_sysfs(self, path: str, value: str) -> bool:
        """Write to a sysfs file."""
        try:
            with open(path, 'w') as f:
                f.write(value)
            return True
        except Exception as e:
            logger.error(f"Failed to write {value} to {path}: {e}")
            return False
    
    @property
    def is_jetson(self) -> bool:
        """Check if running on Jetson."""
        return self._is_jetson
    
    @property
    def model(self) -> JetsonModel:
        """Get Jetson model."""
        return self._model or JetsonModel.UNKNOWN
    
    def get_capabilities(self) -> JetsonCapabilities:
        """
        Get device capabilities.
        
        Returns:
            JetsonCapabilities with device specifications
        """
        if self._capabilities is not None:
            return self._capabilities
        
        model = self.model
        specs = self.MODEL_SPECS.get(model, {})
        
        # Get version info
        jetpack = self._get_jetpack_version()
        cuda = self._get_cuda_version()
        trt = self._get_tensorrt_version()
        
        # Get current power mode
        power_mode = self.get_power_mode()
        available_modes = self._get_available_power_modes()
        
        # Get memory info
        ram_gb = self._get_total_ram_gb()
        swap_gb = self._get_swap_gb()
        
        self._capabilities = JetsonCapabilities(
            model=model,
            jetpack_version=jetpack,
            cuda_version=cuda,
            tensorrt_version=trt,
            gpu_cores=specs.get("gpu_cores", 0),
            dla_cores=specs.get("dla_cores", 0),
            cpu_cores=specs.get("cpu_cores", os.cpu_count() or 4),
            cpu_freq_mhz=self._get_cpu_freq_mhz(),
            ram_gb=ram_gb,
            swap_gb=swap_gb,
            max_power_watts=specs.get("max_power", 15),
            current_power_mode=power_mode,
            available_power_modes=available_modes,
            nvdec_instances=2,  # Typical for Orin
            nvenc_instances=1,
            max_decode_streams=specs.get("max_decode_streams", 16),
        )
        
        return self._capabilities
    
    def _get_jetpack_version(self) -> str:
        """Get JetPack version."""
        # Try apt package version
        output = self._run_command(["apt", "show", "nvidia-jetpack"])
        if output:
            for line in output.split('\n'):
                if line.startswith("Version:"):
                    return line.split(':')[1].strip()
        
        # Try L4T version
        release_file = Path("/etc/nv_tegra_release")
        if release_file.exists():
            try:
                content = release_file.read_text()
                match = re.search(r'R(\d+).*REVISION:\s*([\d.]+)', content)
                if match:
                    return f"L4T {match.group(1)}.{match.group(2)}"
            except Exception:
                pass
        
        return "Unknown"
    
    def _get_cuda_version(self) -> str:
        """Get CUDA version."""
        output = self._run_command(["nvcc", "--version"])
        if output:
            match = re.search(r'release ([\d.]+)', output)
            if match:
                return match.group(1)
        return "Unknown"
    
    def _get_tensorrt_version(self) -> str:
        """Get TensorRT version."""
        try:
            output = self._run_command(["dpkg", "-l", "tensorrt"])
            if output:
                for line in output.split('\n'):
                    if "tensorrt" in line.lower():
                        parts = line.split()
                        if len(parts) >= 3:
                            return parts[2]
        except Exception:
            pass
        return "Unknown"
    
    def _get_swap_gb(self) -> float:
        """Get swap space in GB."""
        try:
            with open("/proc/meminfo", 'r') as f:
                for line in f:
                    if line.startswith("SwapTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        except Exception:
            pass
        return 0
    
    def _get_cpu_freq_mhz(self) -> int:
        """Get CPU frequency in MHz."""
        path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
        value = self._read_sysfs(path)
        if value:
            try:
                return int(value) // 1000  # kHz to MHz
            except ValueError:
                pass
        return 0
    
    def _get_available_power_modes(self) -> List[PowerMode]:
        """Get available power modes."""
        output = self._run_command(["nvpmodel", "-q"])
        if output:
            modes = []
            for line in output.split('\n'):
                for mode in PowerMode:
                    if mode.value in line:
                        modes.append(mode)
            return modes if modes else [PowerMode.MAXN]
        
        # Default modes based on model
        if self.model in [JetsonModel.AGX_ORIN_64GB, JetsonModel.AGX_ORIN_32GB]:
            return [PowerMode.MAXN, PowerMode.MODE_15W]
        elif self.model in [JetsonModel.ORIN_NX_16GB, JetsonModel.ORIN_NX_8GB]:
            return [PowerMode.MAXN, PowerMode.MODE_15W, PowerMode.MODE_10W]
        else:
            return [PowerMode.MAXN, PowerMode.MODE_10W, PowerMode.MODE_5W]
    
    def get_power_mode(self) -> PowerMode:
        """Get current power mode."""
        output = self._run_command(["nvpmodel", "-q"])
        if output:
            for mode in PowerMode:
                if mode.value in output:
                    return mode
        return PowerMode.MAXN
    
    def set_power_mode(self, mode: PowerMode) -> bool:
        """
        Set power mode.
        
        Args:
            mode: Desired power mode
            
        Returns:
            True if successful
        """
        if not self._is_jetson:
            logger.warning("Cannot set power mode: not on Jetson")
            return False
        
        # Map mode to nvpmodel ID
        mode_map = {
            PowerMode.MAXN: 0,
            PowerMode.MODE_15W: 2,
            PowerMode.MODE_10W: 3,
            PowerMode.MODE_7W: 4,
            PowerMode.MODE_5W: 5,
            PowerMode.MODE_2W: 6,
        }
        
        mode_id = mode_map.get(mode, 0)
        
        result = self._run_command(["sudo", "nvpmodel", "-m", str(mode_id)])
        if result is not None:
            logger.info(f"Power mode set to {mode.value}")
            return True
        
        logger.error(f"Failed to set power mode to {mode.value}")
        return False
    
    def get_thermal_status(self) -> ThermalStatus:
        """
        Get thermal status.
        
        Returns:
            ThermalStatus with temperatures and fan info
        """
        timestamp = time.time()
        
        # Read temperatures
        cpu_temp = self._read_temperature("cpu")
        gpu_temp = self._read_temperature("gpu")
        soc_temp = self._read_temperature("soc")
        
        # Read fan status
        fan_speed, fan_pwm = self._read_fan_status()
        
        # Check throttling
        is_throttled, reason = self._check_throttling()
        
        return ThermalStatus(
            timestamp=timestamp,
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            soc_temp=soc_temp,
            fan_speed_percent=fan_speed,
            fan_pwm=fan_pwm,
            is_throttled=is_throttled,
            throttle_reason=reason,
        )
    
    def _read_temperature(self, zone: str) -> float:
        """Read temperature from thermal zone."""
        # Try tegrastats output first
        output = self._run_command(["tegrastats", "--interval", "1", "--stop"])
        if output:
            if zone == "cpu":
                match = re.search(r'CPU@([\d.]+)C', output)
            elif zone == "gpu":
                match = re.search(r'GPU@([\d.]+)C', output)
            else:
                match = re.search(r'SOC@([\d.]+)C|tboard@([\d.]+)C', output)
            
            if match:
                try:
                    return float(match.group(1) or match.group(2))
                except (ValueError, IndexError):
                    pass
        
        # Try sysfs thermal zones
        thermal_base = Path("/sys/devices/virtual/thermal")
        for zone_dir in thermal_base.glob("thermal_zone*"):
            type_path = zone_dir / "type"
            temp_path = zone_dir / "temp"
            
            if type_path.exists() and temp_path.exists():
                try:
                    zone_type = type_path.read_text().strip().lower()
                    if zone in zone_type or zone_type in zone:
                        temp_milli = int(temp_path.read_text().strip())
                        return temp_milli / 1000.0
                except Exception:
                    pass
        
        return 0.0
    
    def _read_fan_status(self) -> Tuple[float, int]:
        """Read fan speed and PWM."""
        # Try to read from pwm-fan
        fan_path = Path("/sys/devices/pwm-fan")
        if fan_path.exists():
            try:
                cur_pwm_path = fan_path / "cur_pwm"
                max_pwm_path = fan_path / "max_pwm"
                
                if cur_pwm_path.exists():
                    cur_pwm = int(cur_pwm_path.read_text().strip())
                    max_pwm = 255
                    
                    if max_pwm_path.exists():
                        max_pwm = int(max_pwm_path.read_text().strip())
                    
                    speed_percent = (cur_pwm / max_pwm) * 100
                    return speed_percent, cur_pwm
            except Exception:
                pass
        
        return 0.0, 0
    
    def _check_throttling(self) -> Tuple[bool, Optional[str]]:
        """Check if device is thermally throttled."""
        # Check CPU throttling
        cpu_path = Path("/sys/devices/system/cpu/cpu0/cpufreq")
        if cpu_path.exists():
            try:
                cur_freq = int((cpu_path / "scaling_cur_freq").read_text().strip())
                max_freq = int((cpu_path / "scaling_max_freq").read_text().strip())
                
                if cur_freq < max_freq * 0.9:  # Running at less than 90% max
                    return True, "CPU thermal throttling"
            except Exception:
                pass
        
        # Check GPU throttling (similar logic)
        # In production, would also check tegrastats for throttling indicators
        
        return False, None
    
    def get_power_status(self) -> PowerStatus:
        """
        Get power consumption status.
        
        Returns:
            PowerStatus with power measurements
        """
        timestamp = time.time()
        power_mode = self.get_power_mode()
        
        # Try INA3221 power monitor
        total_power, gpu_power, cpu_power, soc_power = 0, 0, 0, 0
        voltage, current = 0, 0
        
        # Try tegrastats for power info
        output = self._run_command(["tegrastats", "--interval", "100", "--stop"])
        if output:
            # Parse power from tegrastats output
            # Format varies by platform
            match = re.search(r'VDD_IN (\d+)mW', output)
            if match:
                total_power = float(match.group(1))
            
            match = re.search(r'VDD_GPU_SOC (\d+)mW', output)
            if match:
                gpu_power = float(match.group(1))
            
            match = re.search(r'VDD_CPU (\d+)mW', output)
            if match:
                cpu_power = float(match.group(1))
        
        return PowerStatus(
            timestamp=timestamp,
            total_power_mw=total_power,
            gpu_power_mw=gpu_power,
            cpu_power_mw=cpu_power,
            soc_power_mw=soc_power,
            voltage_mv=voltage,
            current_ma=current,
            power_mode=power_mode,
        )
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get performance metrics.
        
        Returns:
            PerformanceMetrics with utilization info
        """
        timestamp = time.time()
        
        # GPU utilization and frequency
        gpu_util, gpu_freq = self._get_gpu_metrics()
        
        # CPU utilization and frequency
        cpu_util = self._get_cpu_utilization()
        cpu_freq = self._get_cpu_freq_mhz()
        
        # Memory
        ram_used, ram_total = self._get_memory_usage()
        swap_used = self._get_swap_usage()
        
        # EMC
        emc_util, emc_freq = self._get_emc_metrics()
        
        return PerformanceMetrics(
            timestamp=timestamp,
            gpu_util_percent=gpu_util,
            gpu_freq_mhz=gpu_freq,
            cpu_util_percent=cpu_util,
            cpu_freq_mhz=cpu_freq,
            ram_used_mb=ram_used,
            ram_total_mb=ram_total,
            swap_used_mb=swap_used,
            emc_util_percent=emc_util,
            emc_freq_mhz=emc_freq,
        )
    
    def _get_gpu_metrics(self) -> Tuple[float, int]:
        """Get GPU utilization and frequency."""
        util = 0.0
        freq = 0
        
        # Try tegrastats
        output = self._run_command(["tegrastats", "--interval", "100", "--stop"])
        if output:
            match = re.search(r'GR3D_FREQ (\d+)%', output)
            if match:
                util = float(match.group(1))
        
        # Try sysfs for frequency
        freq_path = Path("/sys/devices/gpu.0/devfreq")
        for subdir in freq_path.glob("*"):
            cur_freq_path = subdir / "cur_freq"
            if cur_freq_path.exists():
                try:
                    freq = int(cur_freq_path.read_text().strip()) // 1000000  # Hz to MHz
                    break
                except Exception:
                    pass
        
        return util, freq
    
    def _get_cpu_utilization(self) -> float:
        """Get CPU utilization percentage."""
        try:
            with open("/proc/stat", 'r') as f:
                line = f.readline()
                fields = line.split()
                idle = int(fields[4])
                total = sum(int(x) for x in fields[1:])
                
                if not hasattr(self, '_prev_idle'):
                    self._prev_idle = idle
                    self._prev_total = total
                    return 0.0
                
                idle_diff = idle - self._prev_idle
                total_diff = total - self._prev_total
                
                self._prev_idle = idle
                self._prev_total = total
                
                if total_diff == 0:
                    return 0.0
                
                return (1 - idle_diff / total_diff) * 100
        except Exception:
            return 0.0
    
    def _get_memory_usage(self) -> Tuple[float, float]:
        """Get memory usage in MB."""
        try:
            with open("/proc/meminfo", 'r') as f:
                content = f.read()
            
            total = 0
            available = 0
            
            for line in content.split('\n'):
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024  # KB to MB
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / 1024
            
            used = total - available
            return used, total
        except Exception:
            return 0, 0
    
    def _get_swap_usage(self) -> float:
        """Get swap usage in MB."""
        try:
            with open("/proc/meminfo", 'r') as f:
                content = f.read()
            
            total = 0
            free = 0
            
            for line in content.split('\n'):
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1]) / 1024
                elif line.startswith("SwapFree:"):
                    free = int(line.split()[1]) / 1024
            
            return total - free
        except Exception:
            return 0
    
    def _get_emc_metrics(self) -> Tuple[float, int]:
        """Get EMC (External Memory Controller) metrics."""
        util = 0.0
        freq = 0
        
        # EMC utilization from tegrastats
        output = self._run_command(["tegrastats", "--interval", "100", "--stop"])
        if output:
            match = re.search(r'EMC_FREQ (\d+)%', output)
            if match:
                util = float(match.group(1))
        
        return util, freq
    
    def set_fan_speed(self, speed_percent: float) -> bool:
        """
        Set fan speed.
        
        Args:
            speed_percent: Fan speed 0-100%
            
        Returns:
            True if successful
        """
        if not self._is_jetson:
            return False
        
        # Calculate PWM value (0-255)
        pwm = int((speed_percent / 100) * 255)
        pwm = max(0, min(255, pwm))
        
        fan_pwm_path = "/sys/devices/pwm-fan/target_pwm"
        return self._write_sysfs(fan_pwm_path, str(pwm))
    
    def enable_jetson_clocks(self) -> bool:
        """Enable maximum clock frequencies."""
        result = self._run_command(["sudo", "jetson_clocks"])
        if result is not None:
            logger.info("Jetson clocks enabled")
            return True
        return False
    
    def create_swap(self, size_gb: float = 4.0, path: str = "/swapfile") -> bool:
        """
        Create swap file for additional memory.
        
        Args:
            size_gb: Swap size in GB
            path: Path for swap file
            
        Returns:
            True if successful
        """
        if Path(path).exists():
            logger.warning(f"Swap file {path} already exists")
            return False
        
        size_mb = int(size_gb * 1024)
        
        commands = [
            f"sudo fallocate -l {size_mb}M {path}",
            f"sudo chmod 600 {path}",
            f"sudo mkswap {path}",
            f"sudo swapon {path}",
        ]
        
        for cmd in commands:
            result = self._run_command(cmd.split())
            if result is None:
                logger.error(f"Failed to execute: {cmd}")
                return False
        
        logger.info(f"Created {size_gb}GB swap at {path}")
        return True


class JetsonMonitor:
    """
    Continuous monitoring of Jetson device.
    
    Collects thermal, power, and performance metrics at regular intervals.
    
    Example:
        >>> monitor = JetsonMonitor(interval_sec=1.0)
        >>> monitor.start()
        >>> time.sleep(10)
        >>> summary = monitor.get_summary()
        >>> monitor.stop()
    """
    
    def __init__(
        self,
        interval_sec: float = 1.0,
        history_size: int = 3600,  # 1 hour at 1 sec intervals
    ):
        """
        Initialize monitor.
        
        Args:
            interval_sec: Sampling interval in seconds
            history_size: Number of samples to keep in history
        """
        self.interval_sec = interval_sec
        self.history_size = history_size
        
        self._device = JetsonDevice()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # History
        self._thermal_history: deque = deque(maxlen=history_size)
        self._power_history: deque = deque(maxlen=history_size)
        self._perf_history: deque = deque(maxlen=history_size)
        
        self._lock = threading.Lock()
    
    def start(self):
        """Start monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Jetson monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Jetson monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                thermal = self._device.get_thermal_status()
                power = self._device.get_power_status()
                perf = self._device.get_performance_metrics()
                
                with self._lock:
                    self._thermal_history.append(thermal)
                    self._power_history.append(power)
                    self._perf_history.append(perf)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(self.interval_sec)
    
    def get_current(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            return {
                "thermal": self._thermal_history[-1].to_dict() if self._thermal_history else None,
                "power": self._power_history[-1].to_dict() if self._power_history else None,
                "performance": self._perf_history[-1].to_dict() if self._perf_history else None,
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics over monitoring period."""
        with self._lock:
            if not self._thermal_history:
                return {}
            
            # Thermal summary
            gpu_temps = [t.gpu_temp for t in self._thermal_history]
            cpu_temps = [t.cpu_temp for t in self._thermal_history]
            
            # Power summary
            power_readings = [p.total_power_mw for p in self._power_history]
            
            # Performance summary
            gpu_utils = [p.gpu_util_percent for p in self._perf_history]
            ram_utils = [p.ram_util_percent for p in self._perf_history]
            
            return {
                "duration_sec": len(self._thermal_history) * self.interval_sec,
                "samples": len(self._thermal_history),
                "thermal": {
                    "gpu_temp_avg": np.mean(gpu_temps) if gpu_temps else 0,
                    "gpu_temp_max": np.max(gpu_temps) if gpu_temps else 0,
                    "cpu_temp_avg": np.mean(cpu_temps) if cpu_temps else 0,
                    "cpu_temp_max": np.max(cpu_temps) if cpu_temps else 0,
                },
                "power": {
                    "avg_power_w": np.mean(power_readings) / 1000 if power_readings else 0,
                    "max_power_w": np.max(power_readings) / 1000 if power_readings else 0,
                },
                "performance": {
                    "gpu_util_avg": np.mean(gpu_utils) if gpu_utils else 0,
                    "gpu_util_max": np.max(gpu_utils) if gpu_utils else 0,
                    "ram_util_avg": np.mean(ram_utils) if ram_utils else 0,
                    "ram_util_max": np.max(ram_utils) if ram_utils else 0,
                },
            }
    
    def clear_history(self):
        """Clear monitoring history."""
        with self._lock:
            self._thermal_history.clear()
            self._power_history.clear()
            self._perf_history.clear()


def optimize_for_inference(
    max_streams: int = 16,
    power_mode: PowerMode = PowerMode.MAXN,
) -> Dict[str, Any]:
    """
    Optimize Jetson for maximum inference performance.
    
    Args:
        max_streams: Target number of video streams
        power_mode: Power mode to set
        
    Returns:
        Optimization results
    """
    device = JetsonDevice()
    results = {"success": True, "actions": []}
    
    if not device.is_jetson:
        logger.warning("Not running on Jetson, skipping optimization")
        return {"success": False, "error": "Not on Jetson device"}
    
    # Set power mode
    if device.set_power_mode(power_mode):
        results["actions"].append(f"Set power mode to {power_mode.value}")
    
    # Enable maximum clocks
    if device.enable_jetson_clocks():
        results["actions"].append("Enabled jetson_clocks")
    
    # Set fan to maximum
    if device.set_fan_speed(100):
        results["actions"].append("Set fan to 100%")
    
    # Check if swap is needed
    caps = device.get_capabilities()
    if caps.ram_gb < 8 and caps.swap_gb < 4:
        logger.info("Low RAM and swap detected, consider creating swap")
        results["recommendations"] = ["Create 4GB swap file for stability"]
    
    # Calculate recommended settings
    recommended_batch = min(16, max_streams)
    recommended_workers = min(4, caps.cpu_cores - 2)
    
    results["recommended_settings"] = {
        "inference_batch_size": recommended_batch,
        "worker_threads": recommended_workers,
        "use_dla": caps.dla_cores > 0,
        "max_decode_streams": min(max_streams, caps.max_decode_streams),
    }
    
    logger.info(f"Optimization complete: {results}")
    return results


# Demonstration
if __name__ == "__main__":
    import numpy as np
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Jetson Utilities Demo")
    print("=" * 60)
    
    # Initialize device
    device = JetsonDevice()
    
    print("\n1. Device Detection")
    print("-" * 40)
    print(f"Is Jetson: {device.is_jetson}")
    print(f"Model: {device.model.value}")
    
    # Get capabilities (works in simulation mode too)
    print("\n2. Device Capabilities")
    print("-" * 40)
    
    # Simulate capabilities for demo
    caps = JetsonCapabilities(
        model=JetsonModel.ORIN_NX_16GB,
        jetpack_version="5.1.2",
        cuda_version="12.2",
        tensorrt_version="8.6.1",
        gpu_cores=1024,
        dla_cores=2,
        cpu_cores=8,
        cpu_freq_mhz=2000,
        ram_gb=16.0,
        swap_gb=8.0,
        max_power_watts=25,
        current_power_mode=PowerMode.MAXN,
        available_power_modes=[PowerMode.MAXN, PowerMode.MODE_15W, PowerMode.MODE_10W],
        nvdec_instances=2,
        nvenc_instances=1,
        max_decode_streams=32,
    )
    
    print(f"Model: {caps.model.value}")
    print(f"JetPack: {caps.jetpack_version}")
    print(f"CUDA: {caps.cuda_version}")
    print(f"TensorRT: {caps.tensorrt_version}")
    print(f"GPU Cores: {caps.gpu_cores}")
    print(f"DLA Cores: {caps.dla_cores}")
    print(f"CPU Cores: {caps.cpu_cores}")
    print(f"RAM: {caps.ram_gb} GB")
    print(f"Max Power: {caps.max_power_watts}W")
    print(f"Max Decode Streams: {caps.max_decode_streams}")
    
    # Simulate thermal status
    print("\n3. Thermal Status (Simulated)")
    print("-" * 40)
    
    thermal = ThermalStatus(
        timestamp=time.time(),
        cpu_temp=45.5,
        gpu_temp=42.0,
        soc_temp=44.0,
        fan_speed_percent=50.0,
        fan_pwm=127,
        is_throttled=False,
        throttle_reason=None,
    )
    
    print(f"CPU Temp: {thermal.cpu_temp}°C")
    print(f"GPU Temp: {thermal.gpu_temp}°C")
    print(f"SOC Temp: {thermal.soc_temp}°C")
    print(f"Fan Speed: {thermal.fan_speed_percent}%")
    print(f"Throttled: {thermal.is_throttled}")
    
    # Simulate power status
    print("\n4. Power Status (Simulated)")
    print("-" * 40)
    
    power = PowerStatus(
        timestamp=time.time(),
        total_power_mw=15000,
        gpu_power_mw=8000,
        cpu_power_mw=5000,
        soc_power_mw=2000,
        voltage_mv=12000,
        current_ma=1250,
        power_mode=PowerMode.MAXN,
    )
    
    print(f"Total Power: {power.total_power_mw/1000:.1f}W")
    print(f"GPU Power: {power.gpu_power_mw/1000:.1f}W")
    print(f"CPU Power: {power.cpu_power_mw/1000:.1f}W")
    print(f"Power Mode: {power.power_mode.value}")
    
    # Simulate performance metrics
    print("\n5. Performance Metrics (Simulated)")
    print("-" * 40)
    
    perf = PerformanceMetrics(
        timestamp=time.time(),
        gpu_util_percent=75.0,
        gpu_freq_mhz=1300,
        cpu_util_percent=45.0,
        cpu_freq_mhz=2000,
        ram_used_mb=8500,
        ram_total_mb=16000,
        swap_used_mb=500,
        emc_util_percent=40.0,
        emc_freq_mhz=3200,
    )
    
    print(f"GPU Utilization: {perf.gpu_util_percent}%")
    print(f"GPU Frequency: {perf.gpu_freq_mhz} MHz")
    print(f"CPU Utilization: {perf.cpu_util_percent}%")
    print(f"RAM Usage: {perf.ram_util_percent:.1f}%")
    print(f"EMC Utilization: {perf.emc_util_percent}%")
    
    # Optimization recommendations
    print("\n6. Optimization Recommendations")
    print("-" * 40)
    
    recommendations = {
        "success": True,
        "actions": [
            "Set power mode to MAXN",
            "Enabled jetson_clocks",
            "Set fan to 100%",
        ],
        "recommended_settings": {
            "inference_batch_size": 16,
            "worker_threads": 4,
            "use_dla": True,
            "max_decode_streams": 16,
        },
    }
    
    print("Actions taken:")
    for action in recommendations["actions"]:
        print(f"  ✓ {action}")
    
    print("\nRecommended settings:")
    for key, value in recommendations["recommended_settings"].items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
