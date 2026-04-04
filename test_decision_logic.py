from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class DecisionResult:
    raw_probability: float
    smoothed_probability: float
    decision: str


class ProbabilityDecisionFilter:
    def __init__(self, threshold: float = 0.40, smooth_window: int = 5):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be between 0 and 1")
        if smooth_window < 1:
            raise ValueError("smooth_window must be at least 1")

        self.threshold = threshold
        self.smooth_window = smooth_window
        self.history = deque(maxlen=smooth_window)

    def update(self, horse_probability: float) -> DecisionResult:
        if not (0.0 <= horse_probability <= 1.0):
            raise ValueError(
                f"horse_probability must be between 0 and 1, got {horse_probability}"
            )

        self.history.append(float(horse_probability))
        smoothed = float(np.mean(self.history))

        decision = "horse_present" if smoothed >= self.threshold else "no_horse"

        return DecisionResult(
            raw_probability=float(horse_probability),
            smoothed_probability=smoothed,
            decision=decision,
        )


def print_decision_step(step_index: int, result: DecisionResult) -> None:
    print(
        f"step={step_index:02d} | "
        f"raw={result.raw_probability:.6f} | "
        f"smoothed={result.smoothed_probability:.6f} | "
        f"decision={result.decision}"
    )


if __name__ == "__main__":
    # Example test stream: starts background-like, then horse-like
    test_probabilities = [
        0.10, 0.15, 0.22, 0.18, 0.20,
        0.35, 0.55, 0.72, 0.88, 0.93,
    ]

    decision_filter = ProbabilityDecisionFilter(
        threshold=0.40,
        smooth_window=5,
    )

    for i, prob in enumerate(test_probabilities, start=1):
        result = decision_filter.update(prob)
        print_decision_step(i, result)