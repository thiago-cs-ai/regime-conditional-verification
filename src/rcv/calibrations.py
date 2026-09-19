"""Map probe logits to agreement probabilities."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibration:
    """Calibrate logits to agreement probabilities with Platt scaling."""

    def __init__(self, C=1.0, max_iter=100, l1_ratio=0.0, dual=False, tol=1e-4,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 solver="lbfgs", random_state=None):
        self._logistic = LogisticRegression(
            C=C, max_iter=max_iter, l1_ratio=l1_ratio, dual=dual, tol=tol,
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
            class_weight=class_weight, solver=solver, random_state=random_state)

    def fit(self, logit_scores, agreement):
        """Fit from row-aligned logits and binary agreement labels."""
        self._logistic.fit(np.asarray(logit_scores, dtype=np.float64).reshape(-1, 1), agreement)
        assert_calibration_increasing(float(self._logistic.coef_[0][0]))
        return self

    def probability(self, logit_scores):
        """Return agreement probabilities for row-aligned logits."""
        column = np.asarray(logit_scores, dtype=np.float64).reshape(-1, 1)
        return self._logistic.predict_proba(column)[:, 1]


def assert_calibration_increasing(slope):
    """Require a positive fitted slope to preserve logit order."""
    if slope <= 0:
        shape = "decreasing" if slope < 0 else "flat"
        raise ValueError(
            f"Fitted calibration is {shape}; slope must be positive to preserve logit order "
            f"(got {slope}).")


CALIBRATION_METHODS = {"platt": PlattCalibration}


def build_calibration(spec):
    """Build a calibration from a method specification."""
    spec = dict(spec)
    method = spec.pop("method")
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method {method!r}; available methods: "
                         f"{sorted(CALIBRATION_METHODS)}.")
    return CALIBRATION_METHODS[method](**spec)
