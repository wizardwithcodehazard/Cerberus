# Contributing to Cerberus

Thank you for your interest in contributing to Cerberus! This guide outlines how to set up your local development environment, run tests, retrain the cost model, and submit contributions.

---

## Development Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/wizardwithcodehazard/Cerberus.git
   cd Cerberus
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   
   # Linux / macOS:
   source .venv/bin/activate
   
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies in Editable Mode:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

---

## Running Tests

Run the test suite locally with `pytest`:
```bash
pytest tests/ -v
```

All 10 tests across Clang AST parsing, native regex parsing, XGBoost prediction, OpenMP/OpenACC pragma transformations, and loop-carried safety hazards must pass before opening a PR.

---

## Retraining the ML Model

If you collect new empirical benchmark datasets or modify feature engineering:
```bash
python scripts/train_model.py --data dataset/dataset_merged.csv --out cerberus/trained_model.pkl
```

To merge multiple new target profile CSVs:
```bash
python scripts/merge_datasets.py
```

---

## Adding a New GPU to the Hardware Lookup Table

If OpenCL is unable to query physical silicon registers on a specific platform, Cerberus falls back to its lookup table:
1. Open [`cerberus/hardware.py`](cerberus/hardware.py).
2. Locate the `GPU_MODEL_SPECS` dictionary.
3. Add an entry mapping the lowercase model substring to `(peak_tflops, device_type, is_unified)`:
   ```python
   "rtx 4090": (82.6, "dgpu", False),
   ```
4. Cite the source (e.g., TechPowerUp GPU database or manufacturer datasheet).

---

## Submitting a Pull Request

1. **One Focus per PR:** Keep PRs focused on a single feature, bug fix, or performance optimization.
2. **Add Tests:** If adding a feature or fixing a bug, include a corresponding unit test in `tests/`.
3. **Check Code Quality:** Ensure no temporary build files or untracked artifacts are included.
4. **CI Gate:** Automated GitHub Actions workflows will run the test matrix across Python 3.9 and 3.11.
