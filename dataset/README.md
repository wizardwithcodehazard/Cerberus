# Cerberus Empirical Datasets

This directory contains empirical wall-clock benchmark measurements across physical CPU and GPU hardware configurations for training and evaluating the Cerberus XGBoost cost model.

---

## Datasets Overview

| File | Rows | Target Architecture | Description |
|---|:---:|---|---|
| [`dataset_merged.csv`](dataset_merged.csv) | **2,318** | Multi-Target (5 platforms) | Master cleaned and calibrated dataset across discrete GPUs, unified APUs, and cloud instances. |
| [`new_merged_dataset.csv`](new_merged_dataset.csv) | **5,200+** | Multi-Target Extended | Extended multi-hardware dataset with additional synthetic parameter variations. |
| [`nvidia_rtx3050.csv`](nvidia_rtx3050.csv) | 421 | NVIDIA RTX 3050 Mobile dGPU | Discrete laptop GPU (PCIe Gen4 x8, GDDR6). |
| [`amd_radeon760m.csv`](amd_radeon760m.csv) | 421 | AMD Radeon 760M (RDNA3 APU) | Zero-copy unified memory integrated GPU (LPDDR5). |
| [`dataset_macos.csv`](dataset_macos.csv) | 421 | AMD Radeon Pro 5300M | macOS discrete GPU target on Apple MacBook hardware. |
| [`colab_dataset.csv`](colab_dataset.csv) | 421 | NVIDIA Tesla T4 (Cloud) | Cloud enterprise discrete GPU instance. |
| [`dataset.csv`](dataset.csv) | 634 | Integrated Baseline APU | Initial baseline measurements. |
| [`dataset_merged_original.csv`](dataset_merged_original.csv) | 2,318 | Pre-sanitization Reference | Raw reference baseline before column schema normalization. |

---

## Retraining

To train the production model on the master merged dataset:
```bash
python scripts/train_model.py --data dataset/dataset_merged.csv --out cerberus/trained_model.pkl
```
