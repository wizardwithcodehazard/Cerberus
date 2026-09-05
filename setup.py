from setuptools import setup, find_packages

setup(
    name="markovlens",
    version="0.1.0",
    description="Explainable ML-Guided GPU Offload Profitability Predictor for C/C++ Loops and OpenMP Target Directives",
    author="Team seeplusplus",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.7",
        "rich>=13.7.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "shap>=0.44.0"
    ],
    entry_points={
        "console_scripts": [
            "cerberus = markovlens.cli:main",
            "markovlens = markovlens.cli:main"
        ]
    },
    python_requires=">=3.9",
)
