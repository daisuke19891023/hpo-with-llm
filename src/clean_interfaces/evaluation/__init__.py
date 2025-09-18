"""Utilities for evaluating model behaviour against golden datasets."""

from .datasets import (
    ClassificationRecord,
    FileSearchRecord,
    GoldenDataset,
    load_default_golden_dataset,
    load_golden_dataset,
)
from .indicators import (
    ClassificationIndicator,
    EvaluationIndicator,
    FileSearchIndicator,
    FileSearchPrediction,
    IndicatorResult,
)
from .judge import (
    HeuristicLLMJudge,
    LLMJudgeProtocol,
    LLMJudgeResult,
    ResponsesAPIJudge,
)
from .metrics import ClassificationMetrics, compute_classification_metrics
from .service import EvaluationService, EvaluationSummary, ScoreContribution

__all__ = [
    "ClassificationIndicator",
    "ClassificationMetrics",
    "ClassificationRecord",
    "EvaluationIndicator",
    "EvaluationService",
    "EvaluationSummary",
    "FileSearchIndicator",
    "FileSearchPrediction",
    "FileSearchRecord",
    "GoldenDataset",
    "HeuristicLLMJudge",
    "IndicatorResult",
    "LLMJudgeProtocol",
    "LLMJudgeResult",
    "ResponsesAPIJudge",
    "ScoreContribution",
    "compute_classification_metrics",
    "load_default_golden_dataset",
    "load_golden_dataset",
]
