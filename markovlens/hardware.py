from dataclasses import dataclass
from typing import Optional
import os
import platform

@dataclass
class HardwareProfile:
    name: str
    device_type: str        # 'dgpu', 'igpu', 'egpu', 'cpu'
    type_code: int          # 0: igpu, 1: dgpu, 2: egpu, 3: cpu
    bus_bandwidth_gbps: float
    peak_tflops: float
    unified_memory: bool

PRESET_PROFILES = {
    "dgpu_rtx3060": HardwareProfile(
        name="NVIDIA GeForce RTX 3060",
        device_type="dgpu",
        type_code=1,
        bus_bandwidth_gbps=16.0,   # PCIe Gen4 x8/x16
        peak_tflops=12.7,
        unified_memory=False
    ),
    "dgpu_rtx4090": HardwareProfile(
        name="NVIDIA GeForce RTX 4090",
        device_type="dgpu",
        type_code=1,
        bus_bandwidth_gbps=32.0,   # PCIe Gen4 x16
        peak_tflops=82.6,
        unified_memory=False
    ),
    "igpu_amd_radeon": HardwareProfile(
        name="AMD Radeon 680M Graphics (RDNA2 iGPU)",
        device_type="igpu",
        type_code=0,
        bus_bandwidth_gbps=51.2,   # Dual-channel LPDDR5 unified memory bus
        peak_tflops=3.8,
        unified_memory=True
    ),
    "igpu_intel_iris": HardwareProfile(
        name="Intel Iris Xe Graphics",
        device_type="igpu",
        type_code=0,
        bus_bandwidth_gbps=64.0,   # Shared LPDDR4x/5 bus (zero-copy)
        peak_tflops=2.1,
        unified_memory=True
    ),
    "egpu_thunderbolt": HardwareProfile(
        name="External GPU (Thunderbolt 3/4)",
        device_type="egpu",
        type_code=2,
        bus_bandwidth_gbps=2.8,    # PCIe over TB3/4
        peak_tflops=10.0,
        unified_memory=False
    )
}

def detect_local_hardware() -> HardwareProfile:
    """Dynamically detects host GPU hardware via Windows WMI/CIM or nvidia-smi."""
    # 1. Check for NVIDIA CUDA presence first
    try:
        import subprocess
        res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and res.stdout.strip():
            gpu_name = res.stdout.strip().split("\n")[0]
            return HardwareProfile(
                name=gpu_name,
                device_type="dgpu",
                type_code=1,
                bus_bandwidth_gbps=16.0,
                peak_tflops=12.7,
                unified_memory=False
            )
    except Exception:
        pass

    # 2. Query Windows Video Controllers via PowerShell CIM / WMI
    try:
        import subprocess
        cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and res.stdout.strip():
            gpu_names = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            for name in gpu_names:
                name_lower = name.lower()
                if "radeon" in name_lower or "amd" in name_lower:
                    return HardwareProfile(
                        name=name,
                        device_type="igpu",
                        type_code=0,
                        bus_bandwidth_gbps=51.2,
                        peak_tflops=3.8,
                        unified_memory=True
                    )
                elif "intel" in name_lower or "iris" in name_lower or "uhd" in name_lower:
                    return HardwareProfile(
                        name=name,
                        device_type="igpu",
                        type_code=0,
                        bus_bandwidth_gbps=64.0,
                        peak_tflops=2.1,
                        unified_memory=True
                    )
                elif "geforce" in name_lower or "nvidia" in name_lower or "rtx" in name_lower or "gtx" in name_lower:
                    return HardwareProfile(
                        name=name,
                        device_type="dgpu",
                        type_code=1,
                        bus_bandwidth_gbps=16.0,
                        peak_tflops=12.7,
                        unified_memory=False
                    )
    except Exception:
        pass

    # Fallback to AMD Radeon iGPU profile
    return PRESET_PROFILES["igpu_amd_radeon"]
