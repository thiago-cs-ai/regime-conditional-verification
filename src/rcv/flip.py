"""Apply threshold-based verdict correction."""

import numpy as np


class FlipRule:
    def __init__(self, threshold):
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Flip threshold must be in (0, 1); got {threshold}.")
        self.threshold = threshold

    def flips(self, probability_of_agreement):
        return probability_of_agreement < self.threshold

    def corrected_verdict(self, verdict, probability_of_agreement):
        return np.where(self.flips(probability_of_agreement), 1 - verdict, verdict)
