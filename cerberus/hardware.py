"""Dynamic Hardware Profiler & Micro-Architectural Discovery Module."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import os
import platform
import subprocess
import ctypes
from ctypes import c_uint32, c_uint64, c_void_p, c_size_t, byref, create_string_buffer, POINTER
import logging

logger = logging.getLogger(__name__)

# ─── Physical Constants & Documented Silicon Defaults ───
# These replace all previously hardcoded magic numbers with named, documented values.

# CPU ISA: FP32 FLOPs per cycle per core (multiply + add counted separately)
AVX2_FMA_FLOPS_PER_CYCLE = 16       # 256-bit FMA: 8 FP32 * 2 ops (fused mul+add)
AVX512_FLOPS_PER_CYCLE = 32         # 512-bit FMA: 16 FP32 * 2 ops
SSE_FLOPS_PER_CYCLE = 8             # 128-bit SSE: 4 FP32 * 2 ops
DEFAULT_FLOPS_PER_CYCLE = 16        # Conservative default (AVX2 assumed)

# PCIe Interconnect theoretical peak bandwidths (GB/s, bidirectional per direction)
PCIE_GEN3_X16_BW_GBPS = 15.75
PCIE_GEN3_X8_BW_GBPS = 7.88
PCIE_GEN4_X16_BW_GBPS = 31.5
PCIE_GEN4_X8_BW_GBPS = 15.75
PCIE_GEN5_X16_BW_GBPS = 63.0
PCIE_GEN5_X8_BW_GBPS = 31.5
PCIE_DEFAULT_BW_GBPS = 15.75        # Conservative fallback: PCIe 3.0 x16 / 4.0 x8
THUNDERBOLT3_BW_GBPS = 2.8          # PCIe over Thunderbolt 3/4

# System memory bandwidth defaults (GB/s)
DDR4_3200_DUAL_CH_BW_GBPS = 51.2    # DDR4-3200 dual-channel (128-bit bus)
DDR5_4800_DUAL_CH_BW_GBPS = 76.8    # DDR5-4800 dual-channel
DDR5_5600_DUAL_CH_BW_GBPS = 89.6    # DDR5-5600 dual-channel
LPDDR5_6400_BW_GBPS = 102.4         # LPDDR5-6400 dual-channel (laptop)
DEFAULT_SYSTEM_RAM_BW_GBPS = 51.2   # Conservative fallback

# GPU Kernel launch overhead (seconds) — vendor-dependent
CUDA_LAUNCH_LATENCY_SEC = 5e-6      # NVIDIA CUDA: ~5 microseconds
ROCM_LAUNCH_LATENCY_SEC = 50e-6     # AMD ROCm/HIP: ~50 microseconds
OPENCL_LAUNCH_LATENCY_SEC = 30e-6   # Generic OpenCL: ~30 microseconds
DEFAULT_LAUNCH_LATENCY_SEC = 20e-6  # Conservative default

# GPU silicon: minimum valid clock frequency from OpenCL (MHz)
MIN_VALID_GPU_CLOCK_MHZ = 100       # Below this, driver is likely reporting wrong

# Minimum valid compute TFLOPS before falling back to lookup table
MIN_VALID_TFLOPS = 0.1

# CPU peak TFLOPS clamp range
MIN_CPU_TFLOPS = 0.05
MAX_CPU_TFLOPS = 4.0


# ─── Comprehensive GPU Model Lookup Table ───
# Used as fallback when OpenCL cannot query live silicon registers.
# Maps substrings of GPU display name (lowercased) to (peak_tflops, device_type, is_unified).
# Sources: manufacturer datasheets and TechPowerUp GPU database.

GPU_MODEL_SPECS: Dict[str, Tuple[float, str, bool]] = {
    # NVIDIA GeForce — Desktop
    "rtx 5090":     (104.8, "dgpu", False),
    "rtx 5080":     (56.2, "dgpu", False),
    "rtx 5070 ti":  (44.1, "dgpu", False),
    "rtx 5070":     (30.7, "dgpu", False),
    "rtx 5060 ti":  (23.0, "dgpu", False),
    "rtx 4090":     (82.6, "dgpu", False),
    "rtx 4080 super": (52.2, "dgpu", False),
    "rtx 4080":     (48.7, "dgpu", False),
    "rtx 4070 ti super": (44.1, "dgpu", False),
    "rtx 4070 ti":  (40.1, "dgpu", False),
    "rtx 4070 super": (35.5, "dgpu", False),
    "rtx 4070":     (29.1, "dgpu", False),
    "rtx 4060 ti":  (22.1, "dgpu", False),
    "rtx 4060":     (15.1, "dgpu", False),
    "rtx 3090 ti":  (40.0, "dgpu", False),
    "rtx 3090":     (35.6, "dgpu", False),
    "rtx 3080 ti":  (34.1, "dgpu", False),
    "rtx 3080":     (29.8, "dgpu", False),
    "rtx 3070 ti":  (21.7, "dgpu", False),
    "rtx 3070":     (20.3, "dgpu", False),
    "rtx 3060 ti":  (16.2, "dgpu", False),
    "rtx 3060":     (12.7, "dgpu", False),
    "rtx 3050":     (9.1, "dgpu", False),
    "rtx 2080 ti":  (13.4, "dgpu", False),
    "rtx 2080 super": (11.2, "dgpu", False),
    "rtx 2080":     (10.1, "dgpu", False),
    "rtx 2070 super": (9.1, "dgpu", False),
    "rtx 2070":     (7.5, "dgpu", False),
    "rtx 2060 super": (7.2, "dgpu", False),
    "rtx 2060":     (6.5, "dgpu", False),
    "gtx 1660 ti":  (5.4, "dgpu", False),
    "gtx 1660 super": (5.0, "dgpu", False),
    "gtx 1660":     (4.7, "dgpu", False),
    "gtx 1650 super": (4.4, "dgpu", False),
    "gtx 1650":     (2.9, "dgpu", False),
    "gtx 1080 ti":  (11.3, "dgpu", False),
    "gtx 1080":     (8.9, "dgpu", False),
    "gtx 1070 ti":  (8.2, "dgpu", False),
    "gtx 1070":     (6.5, "dgpu", False),
    "gtx 1060":     (4.4, "dgpu", False),
    "gtx 1050 ti":  (2.1, "dgpu", False),
    # NVIDIA GeForce — Laptop (lower clocks)
    "rtx 4090 laptop": (60.0, "dgpu", False),
    "rtx 4080 laptop": (36.0, "dgpu", False),
    "rtx 4070 laptop": (21.0, "dgpu", False),
    "rtx 4060 laptop": (12.8, "dgpu", False),
    "rtx 4050 laptop": (9.0, "dgpu", False),
    "rtx 3080 laptop": (19.0, "dgpu", False),
    "rtx 3070 laptop": (14.0, "dgpu", False),
    "rtx 3060 laptop": (10.9, "dgpu", False),
    "rtx 3050 laptop": (7.6, "dgpu", False),
    # NVIDIA Workstation / Data Center
    "a100":         (19.5, "dgpu", False),
    "a6000":        (38.7, "dgpu", False),
    "h100":         (51.2, "dgpu", False),
    "t4":           (8.1, "dgpu", False),
    "v100":         (14.0, "dgpu", False),
    # AMD Radeon — Desktop dGPU
    "rx 9070 xt":   (28.3, "dgpu", False),
    "rx 9070":      (22.5, "dgpu", False),
    "rx 7900 xtx":  (61.4, "dgpu", False),
    "rx 7900 xt":   (51.5, "dgpu", False),
    "rx 7800 xt":   (37.3, "dgpu", False),
    "rx 7700 xt":   (35.2, "dgpu", False),
    "rx 7600 xt":   (22.6, "dgpu", False),
    "rx 7600":      (21.5, "dgpu", False),
    "rx 6950 xt":   (23.6, "dgpu", False),
    "rx 6900 xt":   (23.0, "dgpu", False),
    "rx 6800 xt":   (20.7, "dgpu", False),
    "rx 6800":      (16.2, "dgpu", False),
    "rx 6700 xt":   (13.2, "dgpu", False),
    "rx 6600 xt":   (10.6, "dgpu", False),
    "rx 6600":      (8.9, "dgpu", False),
    "rx 6500 xt":   (5.8, "dgpu", False),
    "rx 580":       (6.2, "dgpu", False),
    "rx 570":       (5.1, "dgpu", False),
    "rx 560":       (2.6, "dgpu", False),
    # AMD Radeon — Integrated GPUs (APU)
    "890m":         (7.6, "igpu", True),
    "880m":         (5.8, "igpu", True),
    "780m":         (4.3, "igpu", True),
    "760m":         (3.5, "igpu", True),
    "680m":         (3.38, "igpu", True),
    "660m":         (2.4, "igpu", True),
    "vega 8":       (1.8, "igpu", True),
    "vega 7":       (1.6, "igpu", True),
    "vega 6":       (1.2, "igpu", True),
    "vega 3":       (0.6, "igpu", True),
    # Intel Arc — Discrete
    "arc b580":     (11.9, "dgpu", False),
    "arc a770":     (17.2, "dgpu", False),
    "arc a750":     (14.7, "dgpu", False),
    "arc a580":     (12.6, "dgpu", False),
    "arc a380":     (4.6, "dgpu", False),
    "arc a310":     (3.3, "dgpu", False),
    # Intel — Integrated GPUs
    "iris xe":      (2.1, "igpu", True),
    "iris plus":    (1.1, "igpu", True),
    "uhd 770":      (0.8, "igpu", True),
    "uhd 730":      (0.4, "igpu", True),
    "uhd 630":      (0.4, "igpu", True),
    "uhd 620":      (0.4, "igpu", True),
}

# Conservative floor when no model match is found
UNKNOWN_GPU_TFLOPS = 2.0


@dataclass
class HardwareProfile:
    name: str
    device_type: str        # 'dgpu', 'igpu', 'egpu', 'cpu'
    type_code: int          # 0: igpu, 1: dgpu, 2: egpu, 3: cpu
    bus_bandwidth_gbps: float
    peak_tflops: float
    unified_memory: bool
    cpu_name: str = "Host Processor"
    cpu_tflops: float = 0.450 # Dynamic host CPU FP32 peak (TFLOPS)

    def get_launch_latency_sec(self) -> float:
        """Returns vendor-appropriate kernel launch latency."""
        name_l = self.name.lower()
        if any(kw in name_l for kw in ("nvidia", "geforce", "rtx", "gtx", "tesla", "quadro")):
            return CUDA_LAUNCH_LATENCY_SEC
        elif any(kw in name_l for kw in ("amd", "radeon", "rx ", "vega", "gfx")):
            return ROCM_LAUNCH_LATENCY_SEC
        elif any(kw in name_l for kw in ("intel", "arc", "iris", "xe", "uhd")):
            return OPENCL_LAUNCH_LATENCY_SEC
        return DEFAULT_LAUNCH_LATENCY_SEC


PRESET_PROFILES = {
    "dgpu_rtx3060": HardwareProfile(
        name="NVIDIA GeForce RTX 3060",
        device_type="dgpu",
        type_code=1,
        bus_bandwidth_gbps=PCIE_GEN4_X8_BW_GBPS,
        peak_tflops=12.7,
        unified_memory=False,
        cpu_name="Generic Host CPU",
        cpu_tflops=0.45
    ),
    "dgpu_rtx4090": HardwareProfile(
        name="NVIDIA GeForce RTX 4090",
        device_type="dgpu",
        type_code=1,
        bus_bandwidth_gbps=PCIE_GEN4_X16_BW_GBPS,
        peak_tflops=82.6,
        unified_memory=False,
        cpu_name="Generic Host CPU",
        cpu_tflops=0.85
    ),
    "igpu_amd_radeon": HardwareProfile(
        name="AMD Radeon 680M Graphics (RDNA2 iGPU)",
        device_type="igpu",
        type_code=0,
        bus_bandwidth_gbps=DEFAULT_SYSTEM_RAM_BW_GBPS,
        peak_tflops=3.38,
        unified_memory=True,
        cpu_name="AMD Ryzen CPU",
        cpu_tflops=0.55
    ),
    "igpu_intel_iris": HardwareProfile(
        name="Intel Iris Xe Graphics",
        device_type="igpu",
        type_code=0,
        bus_bandwidth_gbps=64.0,
        peak_tflops=2.1,
        unified_memory=True,
        cpu_name="Intel Core CPU",
        cpu_tflops=0.40
    ),
    "egpu_thunderbolt": HardwareProfile(
        name="External GPU (Thunderbolt 3/4)",
        device_type="egpu",
        type_code=2,
        bus_bandwidth_gbps=THUNDERBOLT3_BW_GBPS,
        peak_tflops=10.0,
        unified_memory=False,
        cpu_name="Generic Host CPU",
        cpu_tflops=0.45
    )
}

def _detect_cpu_isa() -> int:
    """Detects CPU SIMD ISA capability and returns FLOPs per cycle per core."""
    try:
        if platform.system() == "Windows":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if res.returncode == 0:
                name = res.stdout.strip().lower()
                # AVX-512 capable: Intel 11th+ Gen server, Xeon Scalable, AMD Zen4+ (Ryzen 7000+)
                if any(kw in name for kw in ("13th gen", "14th gen", "i9-13", "i9-14",
                                              "sapphire", "emerald", "xeon w-3", "xeon w-2")):
                    return AVX512_FLOPS_PER_CYCLE
                if any(kw in name for kw in ("7945", "7950", "9950", "9900",
                                              "9700", "9600", "genoa", "bergamo")):
                    return AVX512_FLOPS_PER_CYCLE
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "flags" in line.lower():
                        if "avx512f" in line.lower():
                            return AVX512_FLOPS_PER_CYCLE
                        elif "avx2" in line.lower():
                            return AVX2_FMA_FLOPS_PER_CYCLE
                        elif "sse4" in line.lower():
                            return SSE_FLOPS_PER_CYCLE
                        break
    except Exception:
        pass
    return DEFAULT_FLOPS_PER_CYCLE


def detect_host_cpu() -> Tuple[str, float]:
    """Dynamically detects host CPU cores, frequency, and estimates peak FP32 TFLOPS."""
    cpu_name = "Host Processor"
    cores = os.cpu_count() or 4
    clock_ghz = 3.2
    
    try:
        if platform.system() == "Windows":
            cmd = ["powershell", "-NoProfile", "-Command", 
                   "Get-CimInstance Win32_Processor | Select-Object -Property Name, NumberOfCores, MaxClockSpeed | ConvertTo-Json"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if res.returncode == 0 and res.stdout.strip():
                import json
                data = json.loads(res.stdout.strip())
                if isinstance(data, list):
                    data = data[0]
                cpu_name = data.get("Name", cpu_name).strip()
                cores = int(data.get("NumberOfCores", cores))
                max_mhz = float(data.get("MaxClockSpeed", 3200))
                clock_ghz = max(1.0, max_mhz / 1000.0)
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass

    flops_per_cycle = _detect_cpu_isa()
    cpu_tflops = round((cores * clock_ghz * flops_per_cycle) / 1000.0, 3)
    cpu_tflops = max(MIN_CPU_TFLOPS, min(cpu_tflops, MAX_CPU_TFLOPS))
    return cpu_name, cpu_tflops

def _resolve_dynamic_gpu_name(raw_name: str) -> str:
    """Dynamically resolves internal driver codenames (e.g. gfx1035) to OS commercial display names."""
    if not raw_name.lower().startswith("gfx"):
        return raw_name
    try:
        cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            for line in res.stdout.strip().splitlines():
                clean = line.strip()
                if clean and "basic" not in clean.lower():
                    return f"{clean} ({raw_name})"
    except Exception:
        pass
    return raw_name

def _get_vendor_shader_multiplier(dev_name: str, raw_name: str) -> int:
    """Determines shader processors / ALUs per Compute Unit / SM by GPU architecture."""
    name_l = dev_name.lower()
    raw_l = raw_name.lower()

    if any(kw in name_l for kw in ("nvidia", "geforce", "rtx", "gtx", "tesla", "quadro")):
        # NVIDIA Turing / Ampere / Ada Lovelace / Hopper = 128 FP32 ALUs per SM
        return 128
    elif any(kw in name_l for kw in ("intel", "arc", "iris", "xe")):
        # Intel Xe / Arc = 16 Vector Engines per subslice / Xe-core
        return 16
    elif "gfx11" in raw_l or any(kw in name_l for kw in ("rx 7", "780m", "760m", "rdna3", "rx 9")):
        # AMD RDNA3 dual-issue compute units (128 ALUs / WGP)
        return 128
    elif "gfx10" in raw_l or any(kw in name_l for kw in ("rx 6", "680m", "660m", "radeon", "amd")):
        # AMD RDNA1/2: 64 Stream Processors per CU
        return 64
    else:
        # Default Khronos baseline
        return 64

def _lookup_gpu_specs(gpu_name: str) -> Optional[Tuple[float, str, bool]]:
    """Looks up GPU model in the comprehensive specs table. Returns (peak_tflops, device_type, is_unified) or None."""
    name_l = gpu_name.lower()
    # Try longest match first (more specific models like "rtx 4070 ti super" before "rtx 4070")
    for model_key in sorted(GPU_MODEL_SPECS.keys(), key=len, reverse=True):
        if model_key in name_l:
            return GPU_MODEL_SPECS[model_key]
    return None

def _query_pcie_bandwidth(gpu_name: str) -> float:
    """Attempts to query PCIe generation and lane width for discrete GPUs."""
    # Try nvidia-smi first (NVIDIA only)
    name_l = gpu_name.lower()
    if any(kw in name_l for kw in ("nvidia", "geforce", "rtx", "gtx")):
        try:
            cmd = ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader,nounits"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                parts = res.stdout.strip().split(",")
                if len(parts) >= 2:
                    gen = int(parts[0].strip())
                    width = int(parts[1].strip())
                    # PCIe bandwidth per lane per direction: Gen3=0.985, Gen4=1.969, Gen5=3.938 GB/s
                    bw_per_lane = {3: 0.985, 4: 1.969, 5: 3.938}.get(gen, 0.985)
                    return round(bw_per_lane * width, 2)
        except Exception:
            pass

    # Heuristic from GPU generation
    if any(kw in name_l for kw in ("rtx 50", "rx 9")):
        return PCIE_GEN5_X16_BW_GBPS
    elif any(kw in name_l for kw in ("rtx 40", "rx 7", "arc a7", "arc b")):
        return PCIE_GEN4_X16_BW_GBPS
    elif any(kw in name_l for kw in ("rtx 30", "rx 6", "arc a3", "arc a5")):
        return PCIE_GEN4_X8_BW_GBPS
    elif any(kw in name_l for kw in ("rtx 20", "gtx 16", "rx 5")):
        return PCIE_GEN3_X16_BW_GBPS
    elif any(kw in name_l for kw in ("gtx 10",)):
        return PCIE_GEN3_X16_BW_GBPS

    return PCIE_DEFAULT_BW_GBPS


def _query_live_opencl_hardware(cpu_name: str, cpu_tflops: float) -> Optional[HardwareProfile]:
    """Queries all live GPU devices across all OpenCL platforms (Intel, NVIDIA, AMD)."""
    try:
        try:
            cl = ctypes.windll.LoadLibrary("OpenCL.dll")
        except Exception:
            cl = ctypes.cdll.LoadLibrary("libOpenCL.so")

        # Khronos Standard OpenCL ABI Constants (Universal across Intel, NVIDIA, AMD, Apple, ARM)
        CL_PLATFORM_NAME = 0x0902
        CL_DEVICE_NAME = 0x102B
        CL_DEVICE_TYPE = 0x1000
        CL_DEVICE_TYPE_GPU = 0x00000004
        CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
        CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C
        CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035

        # Get all platforms (e.g. Intel OpenCL + NVIDIA CUDA + AMD ROCm)
        num_platforms = c_uint32()
        if cl.clGetPlatformIDs(0, None, byref(num_platforms)) != 0 or num_platforms.value == 0:
            return None

        platform_array = (c_void_p * num_platforms.value)()
        cl.clGetPlatformIDs(num_platforms.value, platform_array, None)

        detected_gpus = []

        for p_id in platform_array:
            num_devices = c_uint32()
            if cl.clGetDeviceIDs(p_id, CL_DEVICE_TYPE_GPU, 0, None, byref(num_devices)) != 0 or num_devices.value == 0:
                continue

            device_array = (c_void_p * num_devices.value)()
            cl.clGetDeviceIDs(p_id, CL_DEVICE_TYPE_GPU, num_devices.value, device_array, None)

            for d_id in device_array:
                # 1. Device Name
                name_buf = create_string_buffer(256)
                cl.clGetDeviceInfo(d_id, CL_DEVICE_NAME, 256, name_buf, None)
                raw_name = name_buf.value.decode('utf-8', errors='ignore').strip()
                dev_name = _resolve_dynamic_gpu_name(raw_name)

                # 2. Max Compute Units
                cu = c_uint32()
                cl.clGetDeviceInfo(d_id, CL_DEVICE_MAX_COMPUTE_UNITS, ctypes.sizeof(cu), byref(cu), None)
                compute_units = max(1, cu.value)

                # 3. Max Clock Frequency (MHz) — use actual reported value
                clk = c_uint32()
                cl.clGetDeviceInfo(d_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, ctypes.sizeof(clk), byref(clk), None)
                clock_mhz = max(MIN_VALID_GPU_CLOCK_MHZ, clk.value)
                if clock_mhz < 500:
                    logger.warning("GPU %s reported clock %d MHz (unusually low)", dev_name, clock_mhz)

                # 4. Host Unified Memory Flag
                unified = c_uint32()
                cl.clGetDeviceInfo(d_id, CL_DEVICE_HOST_UNIFIED_MEMORY, ctypes.sizeof(unified), byref(unified), None)
                
                # Check vendor-neutral discrete vs unified topology
                name_lower = dev_name.lower()
                is_discrete = bool(
                    any(kw in name_lower for kw in ("geforce", "nvidia", "rtx", "gtx", "quadro", "tesla")) or
                    (unified.value == 0 and not any(kw in name_lower for kw in ("radeon", "iris", "uhd", "vega")))
                )
                is_unified = not is_discrete

                # Vendor-aware Peak TFLOPS calculation from live silicon registers
                shaders_per_cu = _get_vendor_shader_multiplier(dev_name, raw_name)
                peak_tflops = round((compute_units * shaders_per_cu * 2 * clock_mhz) / 1e6, 2)
                
                # Validate against lookup table — if OpenCL-computed value is suspiciously low
                if peak_tflops < MIN_VALID_TFLOPS:
                    lookup = _lookup_gpu_specs(dev_name)
                    if lookup:
                        peak_tflops = lookup[0]
                        logger.info("Used lookup table for %s: %.2f TFLOPS", dev_name, peak_tflops)
                    else:
                        peak_tflops = UNKNOWN_GPU_TFLOPS
                        logger.warning("Cannot determine compute capacity for %s, using conservative %.1f TFLOPS", dev_name, peak_tflops)

                if is_unified:
                    ram_bw = _query_live_system_ram_bandwidth()
                    bus_bandwidth = ram_bw if ram_bw > 0 else DEFAULT_SYSTEM_RAM_BW_GBPS
                    dev_type = "igpu"
                    type_code = 0
                else:
                    bus_bandwidth = _query_pcie_bandwidth(dev_name)
                    dev_type = "dgpu"
                    type_code = 1

                detected_gpus.append(HardwareProfile(
                    name=dev_name,
                    device_type=dev_type,
                    type_code=type_code,
                    bus_bandwidth_gbps=bus_bandwidth,
                    peak_tflops=peak_tflops,
                    unified_memory=is_unified,
                    cpu_name=cpu_name,
                    cpu_tflops=cpu_tflops
                ))

        if not detected_gpus:
            return None

        # Prioritize discrete high-performance GPU over integrated if multiple exist
        detected_gpus.sort(key=lambda x: (x.type_code == 1, x.peak_tflops), reverse=True)
        return detected_gpus[0]

    except Exception:
        return None

def _query_live_system_ram_bandwidth() -> float:
    """Queries live physical RAM clock speed and bus width via Windows WMI."""
    try:
        cmd = ["powershell", "-NoProfile", "-Command", 
               "Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property ConfiguredClockSpeed -Average | Select-Object -ExpandProperty Average"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            speed_mhz = float(res.stdout.strip().splitlines()[0])
            # Dual channel 64-bit bus (128 bits total = 16 bytes per cycle)
            bandwidth_gbps = round((speed_mhz * 1e6 * 16) / 1e9, 1)
            if 10.0 <= bandwidth_gbps <= 200.0:
                return bandwidth_gbps
    except Exception:
        pass
    return DEFAULT_SYSTEM_RAM_BW_GBPS

def detect_local_hardware() -> HardwareProfile:
    """Dynamically detects host GPU hardware directly from live silicon registers across all vendors."""
    cpu_name, cpu_tflops = detect_host_cpu()

    # 1. Direct Multi-Platform Driver Query via OpenCL
    live_profile = _query_live_opencl_hardware(cpu_name, cpu_tflops)
    if live_profile:
        return live_profile

    # 2. Query Windows Video Controllers via PowerShell CIM / WMI
    try:
        cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            gpu_names = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            for name in gpu_names:
                if "basic" in name.lower() or "microsoft" in name.lower():
                    continue

                # Look up specs from comprehensive model table
                specs = _lookup_gpu_specs(name)
                if specs:
                    peak_tflops, dev_type, is_unified = specs
                    if is_unified:
                        ram_bw = _query_live_system_ram_bandwidth()
                        bus_bw = ram_bw if ram_bw > 0 else DEFAULT_SYSTEM_RAM_BW_GBPS
                    else:
                        bus_bw = _query_pcie_bandwidth(name)
                    
                    type_code = 0 if dev_type == "igpu" else (2 if dev_type == "egpu" else 1)
                    return HardwareProfile(
                        name=name,
                        device_type=dev_type,
                        type_code=type_code,
                        bus_bandwidth_gbps=bus_bw,
                        peak_tflops=peak_tflops,
                        unified_memory=is_unified,
                        cpu_name=cpu_name,
                        cpu_tflops=cpu_tflops
                    )

                # No model match — classify by vendor with conservative estimate
                name_lower = name.lower()
                if any(kw in name_lower for kw in ("geforce", "nvidia", "rtx", "gtx")):
                    pcie_bw = _query_pcie_bandwidth(name)
                    return HardwareProfile(
                        name=name, device_type="dgpu", type_code=1,
                        bus_bandwidth_gbps=pcie_bw, peak_tflops=UNKNOWN_GPU_TFLOPS,
                        unified_memory=False, cpu_name=cpu_name, cpu_tflops=cpu_tflops
                    )
                elif any(kw in name_lower for kw in ("radeon", "amd")):
                    ram_bw = _query_live_system_ram_bandwidth()
                    return HardwareProfile(
                        name=name, device_type="igpu", type_code=0,
                        bus_bandwidth_gbps=ram_bw if ram_bw > 0 else DEFAULT_SYSTEM_RAM_BW_GBPS,
                        peak_tflops=UNKNOWN_GPU_TFLOPS, unified_memory=True,
                        cpu_name=cpu_name, cpu_tflops=cpu_tflops
                    )
                elif any(kw in name_lower for kw in ("intel", "iris", "uhd", "xe")):
                    return HardwareProfile(
                        name=name, device_type="igpu", type_code=0,
                        bus_bandwidth_gbps=64.0, peak_tflops=UNKNOWN_GPU_TFLOPS,
                        unified_memory=True, cpu_name=cpu_name, cpu_tflops=cpu_tflops
                    )
    except Exception:
        pass

    # Fallback default with detected host CPU
    logger.warning("No GPU detected, using fallback profile")
    return HardwareProfile(
        name="Unknown GPU (Fallback)",
        device_type="igpu",
        type_code=0,
        bus_bandwidth_gbps=DEFAULT_SYSTEM_RAM_BW_GBPS,
        peak_tflops=UNKNOWN_GPU_TFLOPS,
        unified_memory=True,
        cpu_name=cpu_name,
        cpu_tflops=cpu_tflops
    )
