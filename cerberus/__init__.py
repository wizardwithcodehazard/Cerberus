"""Cerberus: Explainable ML-Guided GPU Offload Profitability Predictor for C and C++."""

__version__ = "0.1.0"

from cerberus.model import ProfitabilityModel, PredictionResult
from cerberus.hardware import HardwareProfile, detect_local_hardware, PRESET_PROFILES
from cerberus.parser import LoopFeature, get_ast_parser, CLoopParser
from cerberus.clang_parser import ClangASTParser
from cerberus.transformer import GPUPragmaTransformer, OpenMPTransformer

__all__ = [
    "__version__",
    "ProfitabilityModel",
    "PredictionResult",
    "HardwareProfile",
    "detect_local_hardware",
    "PRESET_PROFILES",
    "LoopFeature",
    "get_ast_parser",
    "ClangASTParser",
    "CLoopParser",
    "GPUPragmaTransformer",
    "OpenMPTransformer",
]
