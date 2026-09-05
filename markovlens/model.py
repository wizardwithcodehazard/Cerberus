"""Neuro-Symbolic ML Cost Model with Roofline Bounds & TreeSHAP Explainability."""

import os
import math
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from markovlens.parser import LoopFeature
from markovlens.hardware import HardwareProfile, PRESET_PROFILES

FEATURE_NAMES = [
    "is_parallel_safe",
    "trip_count",
    "nesting_depth",
    "flops_per_iter",
    "total_flops",
    "memory_footprint_bytes",
    "arithmetic_intensity",
    "data_reuse_ratio",
    "coalescing_efficiency",
    "stride_regularity",
    "branch_divergence_count",
    "has_reduction",
    "hw_type_code",
    "bus_bandwidth_gbps",
    "peak_tflops",
    "unified_memory"
]

FEATURE_LABELS = {
    "is_parallel_safe": "Loop-Carried Parallel Safety",
    "trip_count": "Loop Trip Count & Parallelism",
    "nesting_depth": "Loop Nesting Depth",
    "flops_per_iter": "Compute FLOPs Per Iteration",
    "total_flops": "Total Compute Workload",
    "memory_footprint_bytes": "Host-Device Transfer Volume",
    "arithmetic_intensity": "Arithmetic Intensity (FLOP/Byte)",
    "data_reuse_ratio": "Temporal/Spatial Data Reuse Ratio",
    "coalescing_efficiency": "SIMD Memory Coalescing Efficiency",
    "stride_regularity": "Memory Access Stride Regularity",
    "branch_divergence_count": "Control Flow Branch Divergence",
    "has_reduction": "Parallel Reduction Accumulator",
    "hw_type_code": "Target GPU Architecture Class",
    "bus_bandwidth_gbps": "Host-Device Interconnect Bandwidth",
    "peak_tflops": "Target GPU Compute Capacity",
    "unified_memory": "Unified Memory Architecture (Zero-Copy)"
}

@dataclass
class RooflineBound:
    peak_gflops: float
    bandwidth_gbps: float
    arithmetic_intensity: float
    attainable_gflops: float
    is_memory_bound: bool

@dataclass
class PredictionResult:
    is_profitable: bool
    predicted_speedup: float
    confidence: float
    roofline: RooflineBound
    primary_explanation: str
    top_positive_factors: List[Tuple[str, float]]
    top_negative_factors: List[Tuple[str, float]]
    shap_values: Dict[str, float]


class ProfitabilityModel:
    """Neuro-symbolic profitability predictor combining Williams Roofline Model with XGBoost."""

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[xgb.XGBRegressor] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.model_path = model_path
        
        default_trained = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        elif os.path.exists(default_trained):
            self.load(default_trained)
        else:
            self._init_bootstrap_model()

    def compute_roofline(self, loop: LoopFeature, hw: HardwareProfile) -> RooflineBound:
        """Computes classical Williams Roofline Model theoretical bound."""
        peak_gflops = hw.peak_tflops * 1000.0
        # Effective bandwidth
        eff_bw = hw.bus_bandwidth_gbps * loop.coalescing_efficiency * loop.stride_regularity
        
        # Roofline formula: Attainable GFLOPS = min(Peak GFLOPS, Arithmetic Intensity * Bandwidth)
        ceiling_from_bw = loop.arithmetic_intensity * eff_bw
        attainable_gflops = min(peak_gflops, ceiling_from_bw)
        is_memory_bound = ceiling_from_bw < peak_gflops

        return RooflineBound(
            peak_gflops=peak_gflops,
            bandwidth_gbps=eff_bw,
            arithmetic_intensity=loop.arithmetic_intensity,
            attainable_gflops=attainable_gflops,
            is_memory_bound=is_memory_bound
        )

    def _init_bootstrap_model(self):
        """Initializes a physically grounded pre-trained XGBoost cost model."""
        np.random.seed(42)
        rows = []
        
        dims = [32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536, 262144, 1048576]
        depths = [1, 2, 3]
        f_per_iters = [1, 2, 4, 16, 32]
        strides = [1.0, 0.5, 0.1]
        branches = [0, 1, 3]
        coalesces = [1.0, 0.4, 0.2]
        hw_profiles = list(PRESET_PROFILES.values())

        for d in dims:
            for depth in depths:
                for f_iter in f_per_iters:
                    for stride in strides:
                        for branch in branches:
                            for coalesce in coalesces:
                                for hw in hw_profiles:
                                    arrays = 3
                                    if depth == 1:
                                        trip = d
                                        elements = d
                                        total_flops = trip * f_iter
                                        reuse = 1.0
                                    elif depth == 2:
                                        trip = d * d
                                        elements = d * d
                                        total_flops = trip * f_iter
                                        reuse = 2.0
                                    else: # depth == 3 (e.g. GEMM)
                                        trip = d * d * d
                                        elements = d * d
                                        total_flops = 2 * d * d * d
                                        reuse = float(d) # O(N) cache reuse

                                    bytes_transferred = max(elements * 4 * arrays, 256)
                                    arith_intensity = total_flops / float(bytes_transferred)
                                    
                                    # Physics model
                                    cpu_tflops = 0.035
                                    t_cpu_sec = total_flops / (cpu_tflops * 1e12)
                                    
                                    launch_latency_sec = 20e-6
                                    if hw.unified_memory:
                                        transfer_sec = 0.0
                                    else:
                                        transfer_sec = bytes_transferred / (hw.bus_bandwidth_gbps * 1e9 * 0.85)
                                    
                                    efficiency = max(0.05, stride * coalesce * (1.0 / (1.0 + branch * 0.4)))
                                    gpu_compute_sec = total_flops / (hw.peak_tflops * 1e12 * efficiency)
                                    
                                    t_gpu_sec = launch_latency_sec + transfer_sec + gpu_compute_sec

                                    speedup = t_cpu_sec / max(t_gpu_sec, 1e-9)
                                    log_speedup = math.log2(max(speedup, 0.001))

                                    rows.append([
                                        1.0, # safe
                                        float(trip), float(depth), float(f_iter), float(total_flops),
                                        float(bytes_transferred), float(arith_intensity), float(reuse),
                                        float(coalesce), float(stride), float(branch), 0.0,
                                        float(hw.type_code), float(hw.bus_bandwidth_gbps),
                                        float(hw.peak_tflops), 1.0 if hw.unified_memory else 0.0,
                                        log_speedup
                                    ])

        df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["target_log_speedup"])
        X = df[FEATURE_NAMES]
        y = df["target_log_speedup"]

        self.model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            random_state=42
        )
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    def predict_loop(self, loop: LoopFeature, hw: HardwareProfile, speedup_threshold: float = 1.1) -> PredictionResult:
        """Predicts GPU offload profitability combining Roofline bounds & TreeSHAP."""
        roofline = self.compute_roofline(loop, hw)

        # 1. Dependency Safety Gate
        if not loop.is_parallel_safe:
            return PredictionResult(
                is_profitable=False,
                predicted_speedup=0.0,
                confidence=1.0,
                roofline=roofline,
                primary_explanation=f"GATING REJECT: {loop.safety_reason}. Offloading would introduce race conditions.",
                top_positive_factors=[],
                top_negative_factors=[("Loop-Carried Parallel Safety", -10.0)],
                shap_values={"is_parallel_safe": -10.0}
            )

        # 2. ML Prediction
        feature_vector = loop.to_feature_vector() + [
            float(hw.type_code),
            float(hw.bus_bandwidth_gbps),
            float(hw.peak_tflops),
            1.0 if hw.unified_memory else 0.0
        ]

        X_sample = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)
        log_speedup_pred = float(self.model.predict(X_sample)[0])
        predicted_speedup = 2.0 ** log_speedup_pred

        is_profitable = predicted_speedup >= speedup_threshold

        # 3. TreeSHAP Computation
        raw_shap = self.explainer.shap_values(X_sample)[0]
        shap_dict = {name: float(val) for name, val in zip(FEATURE_NAMES, raw_shap)}

        pos_factors = sorted([(k, v) for k, v in shap_dict.items() if v > 0], key=lambda x: x[1], reverse=True)
        neg_factors = sorted([(k, v) for k, v in shap_dict.items() if v < 0], key=lambda x: x[1])

        explanation = self._generate_explanation(is_profitable, predicted_speedup, shap_dict, loop, hw, roofline)
        confidence = min(0.99, max(0.55, abs(log_speedup_pred) / (abs(log_speedup_pred) + 1.0)))

        return PredictionResult(
            is_profitable=is_profitable,
            predicted_speedup=predicted_speedup,
            confidence=confidence,
            roofline=roofline,
            primary_explanation=explanation,
            top_positive_factors=[(FEATURE_LABELS.get(k, k), v) for k, v in pos_factors[:3]],
            top_negative_factors=[(FEATURE_LABELS.get(k, k), v) for k, v in neg_factors[:3]],
            shap_values=shap_dict
        )

    def _generate_explanation(self, is_profitable: bool, speedup: float, shap_dict: Dict[str, float],
                              loop: LoopFeature, hw: HardwareProfile, roofline: RooflineBound) -> str:
        bound_desc = "Memory-Bound (limited by bus bandwidth)" if roofline.is_memory_bound else "Compute-Bound (saturating GPU ALUs)"
        
        if is_profitable:
            reasons = []
            if loop.data_reuse_ratio > 4.0:
                reasons.append(f"high data reuse ({loop.data_reuse_ratio:.1f}x) effectively utilizes GPU cache")
            if shap_dict.get("trip_count", 0) > 0.15:
                reasons.append(f"abundant parallelism ({loop.trip_count:,} iterations)")
            if loop.coalescing_efficiency > 0.8:
                reasons.append("optimal row-major SIMD coalescing")
            if not reasons:
                reasons.append(f"high compute throughput on {hw.name}")
            return f"GPU OFFLOAD PROFITABLE ({speedup:.2f}x speedup | Roofline: {roofline.attainable_gflops:.1f} GFLOPS, {bound_desc}). " + " and ".join(reasons) + "."
        else:
            reasons = []
            if shap_dict.get("memory_footprint_bytes", 0) < -0.15:
                reasons.append(f"host-device transfer overhead ({loop.memory_footprint_bytes / 1024:.1f} KB) dominates compute")
            if shap_dict.get("trip_count", 0) < -0.15:
                reasons.append(f"small iteration count ({loop.trip_count}) cannot amortize launch latency")
            if loop.coalescing_efficiency < 0.5:
                reasons.append("non-coalesced strided memory access degrades warp bandwidth")
            if shap_dict.get("branch_divergence_count", 0) < -0.15:
                reasons.append("control flow branching causes warp divergence")
            if not reasons:
                reasons.append(f"overhead on {hw.name} exceeds sequential CPU baseline")
            return f"KEEP CPU SEQUENTIAL (GPU predicted at {speedup:.2f}x slowdown | Roofline: {bound_desc}). " + " and ".join(reasons) + "."

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({"model": self.model}, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.explainer = shap.TreeExplainer(self.model)
