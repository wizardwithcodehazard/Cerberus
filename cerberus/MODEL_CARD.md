# Model Card: Cerberus Cost Model (`trained_model.pkl`)

## Model Summary

- **Model Name:** Cerberus Two-Stage Physics-Constrained Hurdle XGBoost Model
- **Primary Task:** Gating and speedup regression for GPU loop offloading (CPU OpenMP vs. GPU OpenCL/OpenMP Target)
- **Artifact:** [`cerberus/trained_model.pkl`](trained_model.pkl)
- **Target Metrics:**
  1. `is_profitable`: Binary classification ($\text{Speedup} \ge 1.05\times$)
  2. `log2_speedup`: Continuous regression ($\log_2(\text{Time}_{\text{CPU}} / \text{Time}_{\text{GPU}})$)

---

## Architecture & Features

The model uses a two-stage hurdle approach with monotonicity constraints on total FLOPs, arithmetic intensity, and transfer overheads:

### Feature Schema (21 Dimensions)
1. `is_parallel_safe`: Loop-carried data dependency / race safety
2. `trip_count`: Total iteration count
3. `nesting_depth`: Outer loop hierarchy depth ($1$, $2$, or $3+$)
4. `flops_per_iter`: Body arithmetic floating-point operations
5. `total_flops`: Total compute workload ($\text{flops\_per\_iter} \times \text{trip\_count}$)
6. `memory_footprint_bytes`: Unique tensor working set volume
7. `arithmetic_intensity`: Workload intensity ($\text{total\_flops} / \text{footprint}$)
8. `data_reuse_ratio`: Working set cache locality factor
9. `coalescing_efficiency`: SIMD stride coalescing rating ($0.2$ to $1.0$)
10. `stride_regularity`: Memory access stride continuity
11. `branch_divergence_count`: Conditional branches inside loop body
12. `has_reduction`: Reduction accumulator detection
13. `hw_type_code`: Device architecture class ($0=\text{iGPU}, 1=\text{dGPU}, 2=\text{eGPU}$)
14. `bus_bandwidth_gbps`: PCIe / Interconnect transfer bandwidth
15. `peak_tflops`: Hardware peak FP32 compute capability
16. `unified_memory`: Zero-copy unified memory flag ($0$ or $1$)
17. `transfer_to_compute_ratio`: Data transfer duration divided by kernel compute time
18. `log2_trip_count`: $\log_2(\max(\text{trip\_count}, 1))$
19. `log2_total_flops`: $\log_2(\max(\text{total\_flops}, 1))$
20. `log2_footprint_bytes`: $\log_2(\max(\text{footprint}, 1))$
21. `roofline_attainable_gflops`: Williams Roofline theoretical throughput bound

---

## Training Datasets

- **Master Benchmark Dataset:** [`dataset/dataset_merged.csv`](../dataset/dataset_merged.csv)  
  Contains 2,318 empirical silicon measurements collected across 5 physical platforms:
  - NVIDIA RTX 3050 Laptop dGPU (421 runs)
  - AMD Radeon 760M RDNA3 APU iGPU (421 runs)
  - Apple macOS AMD Radeon Pro 5300M (421 runs)
  - Google Colab NVIDIA Tesla T4 Cloud GPU (421 runs)
  - Intel / AMD Integrated APU Baseline (634 runs)
- **Extended Dataset:** [`dataset/new_merged_dataset.csv`](../dataset/new_merged_dataset.csv)  
  Contains extended multi-hardware measurements and parametric variations.

---

## Cross-Validation Performance (Stratified 5-Fold)

| Metric | Score (Mean ± Std) |
|---|:---:|
| **ROC-AUC** | **`0.9669 ± 0.0065`** |
| **Gating Accuracy** | **`91.33% ± 1.02%`** |
| **Precision** | **`86.74% ± 3.51%`** |
| **Recall** | **`68.69% ± 4.43%`** |
| **F1-Score** | **`0.7655 ± 0.0300`** |
| **$R^2$ Score** | **`0.8362 ± 0.0310`** |
| **RMSE** | **`1.1765 ± 0.0838`** |

---

## Reproducibility & Retraining

To retrain the production model artifact:
```bash
python scripts/train_model.py --data dataset/dataset_merged.csv --out cerberus/trained_model.pkl
```
