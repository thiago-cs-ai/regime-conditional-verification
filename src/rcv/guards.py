"""Shared regime vocabulary and pre-fit validation guards."""

import numpy as np

REGIMES = (0, 1)
POOLED = "pooled"


class MissingCaseError(ValueError):
    """A fitting slice lacks a regime-agreement case."""


class ScoreIntegrityError(ValueError):
    """Scores fail a required numerical integrity check."""


PINNED_ROWS_THAT_SIGNAL_A_BOUND = 3


def assert_scores_rankable(scores, context, inputs=None, check_pinning=True):
    """Require float64, finite scores without a repeated extreme value."""
    if scores.dtype != np.float64:
        raise ScoreIntegrityError(f"{context}: score dtype must be float64; got {scores.dtype}.")
    if not np.isfinite(scores).all():
        raise ScoreIntegrityError(f"{context}: scores contain non-finite values.")
    if not check_pinning:
        return scores
    for bound in (scores.min(), scores.max()):
        pinned_rows = scores == bound
        pinned = int(pinned_rows.sum())
        if pinned >= PINNED_ROWS_THAT_SIGNAL_A_BOUND:
            if inputs is not None:
                pinned = len(np.unique(np.asarray(inputs)[pinned_rows], axis=0, equal_nan=True))
                if pinned < PINNED_ROWS_THAT_SIGNAL_A_BOUND:
                    continue
            raise ScoreIntegrityError(f"{context}: {pinned} items are pinned at extreme score "
                                      f"{float(bound)!r}.")
    return scores


def assert_regimes(verdict, context):
    """Require binary verdicts for regime routing."""
    if not np.isin(verdict, REGIMES).all():
        raise ValueError(f"{context}: verdict contains values outside regimes {REGIMES}.")
    return verdict


def assert_all_four_cases(fitting_slice, slice_name):
    """Require every verdict-agreement combination in a fitting slice."""
    for regime in REGIMES:
        for agrees in (0, 1):
            if not np.any((fitting_slice.verdict == regime) & (fitting_slice.agreement == agrees)):
                raise MissingCaseError(f"{slice_name} slice lacks regime={regime}, "
                                       f"agreement={agrees}.")
