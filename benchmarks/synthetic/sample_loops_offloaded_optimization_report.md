# Cerberus Compiler Optimization & GPU Profitability Report

**Source File:** `benchmarks/synthetic/sample_loops.c`  
**Target Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (DGPU)  
**Host Processor:** AMD Ryzen 5 8645HS w/ Radeon 760M Graphics (~0.413 TFLOPS FP32 Peak)  
**Memory Bandwidth:** 15.75 GB/s (PCIe Interconnect)  
**Compute Capacity:** 6.03 TFLOPS  
**Target Dialect:** `OPENMP` (Offload pragmas)  
**Cost Model:** XGBoost Regressor + TreeSHAP (Trained on 2,318 silicon runs -- CV ROC-AUC: 0.967, R2: 0.836)

## Executive Summary
- **Total Loop Regions Analyzed:** 3
- **GPU Offload Injected (Profitable):** 0 regions
- **CPU Sequential Preserved (Slowdowns Prevented):** 3 regions
- **Unsafe Race Hazards Blocked:** 0 regions

## Loop Optimization Gating Table

| Region | Depth | Dynamic Iterations | Effective AI | Roofline Attainable | Predicted Speedup (95% CI) | Gating Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| #1 `vector_add_small()` (L5-7) | 1 | 256 | 0.08 FLOP/B | 1.3 GFLOPS | 0.19x [0.0x-1.0x] | KEEP CPU |
| #2 `matmul_dense()` (L12-20) | 3 | 1,073,741,824 | 0.17 FLOP/B | 2.0 GFLOPS | 14.41x [2.9x-71.3x] | KEEP CPU |
| #3 `branchy_stencil()` (L25-31) | 1 | 10,000 | 0.62 FLOP/B | 8.3 GFLOPS | 0.42x [0.1x-2.1x] | KEEP CPU |

## Detailed Loop-by-Loop Micro-Architectural Audits

### Loop #1: `vector_add_small()` (Lines 5-7)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.19x slowdown [95% CI: 0.04x-0.95x] | Roofline Ceiling: 1.3 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Total Compute Workload (-0.65 SHAP), Loop Trip Count & Parallelism (-0.61 SHAP).
- **Speedup Estimate (95% CI):** 0.19x (Range: 0.04x - 0.95x | Confidence: 70%)
- **Workload:** 256 iterations | 0.00 MFLOPs (1 FLOP/iter)
- **Memory Working Set:** 0.00 MB (21.3x estimated cache reuse)
- **SIMD Coalescing Score:** 100% (Stride regularity: 1.00)
- **Theoretical Roofline Ceiling:** 1.3 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+1.850 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.198 SHAP` : Log2 Scaled Total Compute Workload
  - `+0.168 SHAP` : Host-Device Interconnect Bandwidth
  - `-0.652 SHAP` : Total Compute Workload
  - `-0.614 SHAP` : Loop Trip Count & Parallelism
  - `-0.361 SHAP` : Loop Nesting Depth

---

### Loop #2: `matmul_dense()` (Lines 12-20)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 14.41x slowdown [95% CI: 2.91x-71.32x] | Roofline Ceiling: 2.0 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Log2 Scaled Total Compute Workload (-0.82 SHAP), Data Movement & Interconnect Overhead (-0.57 SHAP).
- **Speedup Estimate (95% CI):** 14.41x (Range: 2.91x - 71.32x | Confidence: 79%)
- **Workload:** 1,073,741,824 iterations | 2147.48 MFLOPs (2 FLOP/iter)
- **Memory Working Set:** 12884.90 MB (89478485.3x estimated cache reuse)
- **SIMD Coalescing Score:** 87% (Stride regularity: 0.87)
- **Theoretical Roofline Ceiling:** 2.0 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+3.939 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+2.278 SHAP` : Loop Nesting Depth
  - `+1.357 SHAP` : Total Compute Workload
  - `-0.818 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.573 SHAP` : Data Movement & Interconnect Overhead
  - `-0.461 SHAP` : Williams Roofline Theoretical Upper Bound

---

### Loop #3: `branchy_stencil()` (Lines 25-31)
- **Parallel Safety:** SAFE
- **Primary Gating Rationale:** KEEP CPU SEQUENTIAL (Predicted: 0.42x slowdown [95% CI: 0.09x-2.08x] | Roofline Ceiling: 8.3 GFLOPS [Interconnect-Constrained]). Sequential execution favored due to Loop Nesting Depth (-0.47 SHAP), Total Compute Workload (-0.45 SHAP).
- **Speedup Estimate (95% CI):** 0.42x (Range: 0.09x - 2.08x | Confidence: 56%)
- **Workload:** 10,000 iterations | 0.05 MFLOPs (5 FLOP/iter)
- **Memory Working Set:** 0.08 MB (2500.0x estimated cache reuse)
- **SIMD Coalescing Score:** 92% (Stride regularity: 0.92)
- **Theoretical Roofline Ceiling:** 8.3 GFLOPS (Bandwidth-Bound)

**TreeSHAP Factor Breakdown:**
  - `+2.065 SHAP` : Temporal/Spatial Cache Data Reuse
  - `+0.223 SHAP` : Host-Device Interconnect Bandwidth
  - `+0.182 SHAP` : Log2 Scaled Total Compute Workload
  - `-0.472 SHAP` : Loop Nesting Depth
  - `-0.454 SHAP` : Total Compute Workload
  - `-0.320 SHAP` : Loop Trip Count & Parallelism

---
