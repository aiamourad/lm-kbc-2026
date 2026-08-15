# The chat-based comparator: the 'it, chat' column of the paper.
from __future__ import annotations

from .decode import Calibrator
from .data import RELATIONS, RELATION_TYPE, Row, load_split, write_predictions
from .decode import DecodeResult, decode, expected_f1_curve
from .propose import RowPrediction, Solver, SolverConfig
from .clients import OpenAICompatibleClient, ResponseCache, check_param_budget

__version__ = "0.1.0"

__all__ = [
    "Calibrator",
    "DecodeResult",
    "RELATIONS",
    "RELATION_TYPE",
    "ResponseCache",
    "Row",
    "RowPrediction",
    "Solver",
    "SolverConfig",
    "OpenAICompatibleClient",
    "check_param_budget",
    "decode",
    "expected_f1_curve",
    "load_split",
    "write_predictions",
]
