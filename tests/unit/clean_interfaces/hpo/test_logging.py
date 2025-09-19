"""Tests for HPO trial logging utilities."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from clean_interfaces.hpo.logging import CSVTrialLogger
from clean_interfaces.hpo.schemas import (
    CodingTask,
    HPOTrialRequest,
    HPOTrialResponse,
    HPOSuggestedTrial,
    TrialObservation,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_csv_logger_writes_trial_rows(tmp_path: Path) -> None:
    """CSV loggers should append rows describing completed trials."""
    log_path = tmp_path / "trials.csv"
    logger = CSVTrialLogger(log_path)

    request = HPOTrialRequest(
        task=CodingTask(task_id="demo", description="Demo"),
        trial=HPOSuggestedTrial(trial_id="1", hyperparameters={"temperature": 0.3}),
        search_space=(),
        history=(),
    )
    response = HPOTrialResponse(score=0.82)
    observation = TrialObservation(
        trial_id="1",
        hyperparameters={"temperature": 0.3},
        score=0.82,
        succeeded=True,
        metadata={
            "evaluation": {
                "dataset": "synthetic",
                "summary": {
                    "indicators": [
                        {
                            "name": "classification",
                            "metrics": {
                                "accuracy": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "f1": 1.0,
                            },
                            "score_components": [1.0, 1.0, 1.0, 1.0],
                        },
                    ],
                    "composite_score": 0.82,
                    "judge": {"score": 0.75},
                },
                "plan": "Operate at low temperature",
                "inputs": {
                    "classification": {
                        "labels": [True],
                        "predictions": [True],
                        "alignment": [],
                    },
                },
            },
        },
        notes="classification benchmark",
    )

    logger.record(request=request, response=response, observation=observation)

    with log_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["trial_id"] == "1"
    assert row["evaluation_indicators"] == "classification"
    assert row["plan_description"] == "Operate at low temperature"
    assert row["dataset"] == "synthetic"
    assert "temperature" in row["parameters"]
