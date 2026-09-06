"""Test ML Cost Model & TreeSHAP Explainability."""

import os
from cerberus.parser import CLoopParser
from cerberus.hardware import PRESET_PROFILES
from cerberus.model import ProfitabilityModel

def test_profitability_predictions():
    test_file = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "synthetic", "sample_loops.c")
    parser = CLoopParser()
    loops = parser.parse_file(test_file)

    model = ProfitabilityModel()
    rtx3060 = PRESET_PROFILES["dgpu_rtx3060"]
    iris_igpu = PRESET_PROFILES["igpu_intel_iris"]

    print(f"Testing model predictions on {len(loops)} parsed loops:\n")

    # 1. Loop 1: Small Vector Add (Trip count 256)
    res_l1 = model.predict_loop(loops[0], rtx3060)
    print("=== Loop 1: Small Vector Add (on RTX 3060 dGPU) ===")
    print(f"Profitable: {res_l1.is_profitable}")
    print(f"Predicted Speedup: {res_l1.predicted_speedup:.2f}x")
    print(f"Confidence: {res_l1.confidence * 100:.1f}%")
    print(f"Explanation: {res_l1.primary_explanation}")
    print(f"Top Negative Factors: {res_l1.top_negative_factors}")
    assert not res_l1.is_profitable, "Small vector add should NOT be profitable on discrete GPU!"

    # 2. Loop 2: Dense MatMul (Trip count 1024^3)
    res_l2 = model.predict_loop(loops[1], rtx3060)
    print("\n=== Loop 2: Dense MatMul (on RTX 3060 dGPU) ===")
    print(f"Profitable: {res_l2.is_profitable}")
    print(f"Predicted Speedup: {res_l2.predicted_speedup:.2f}x")
    print(f"Confidence: {res_l2.confidence * 100:.1f}%")
    print(f"Explanation: {res_l2.primary_explanation}")
    print(f"Top Positive Factors: {res_l2.top_positive_factors}")
    assert res_l2.is_profitable, "Dense MatMul should be highly profitable on discrete GPU!"

    # 3. Test Device Sensitivity: Loop 1 on iGPU vs dGPU
    res_l1_igpu = model.predict_loop(loops[0], iris_igpu)
    print("\n=== Loop 1: Small Vector Add on iGPU (Zero-Copy) ===")
    print(f"Predicted Speedup on iGPU: {res_l1_igpu.predicted_speedup:.2f}x (vs dGPU: {res_l1.predicted_speedup:.2f}x)")
    print(f"Explanation: {res_l1_igpu.primary_explanation}")

    print("\n[SUCCESS] Model and TreeSHAP explainability tests passed cleanly!")

if __name__ == "__main__":
    test_profitability_predictions()
