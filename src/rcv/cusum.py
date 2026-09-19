"""CUSUM increments, noise-boundary shifts, and reachability checks."""

import numpy as np
from scipy import optimize, stats

MINIMUM_STREAMS_IN_THE_TAIL = 10

JEFFREYS_PRIOR_WEIGHT = 0.5

AT_REFERENCE_REPLAY_BATCH = 2**31


def at_reference_replay_seed(seed):
    """Return a deterministic substream seed for at-reference replay."""
    return [*seed, AT_REFERENCE_REPLAY_BATCH] if isinstance(seed, (list, tuple)) else [
        seed, AT_REFERENCE_REPLAY_BATCH]


SHIFT_SPECIFICATION_KEYS = {"indifference_zone_quantile"}

LARGEST_SHIFT_FRACTION = 1.0 - 1e-9
SMALLEST_SHIFT = 1e-12
SHIFT_ROOT_TOLERANCE = 1e-15
REACHABILITY_TOLERANCE = 1e-4


class ThresholdOutOfReach(ValueError):
    """A calibrated threshold cannot be reached within the horizon."""

    def __init__(self, key):
        super().__init__(f"No admissible shift reaches the threshold for {key}.")
        self.key = key


def assert_shift_specification(detectable_shift):
    """Validate the indifference-zone shift specification."""
    if not isinstance(detectable_shift, dict):
        raise ValueError(
            "detectable_shift must be a mapping with key 'indifference_zone_quantile'; "
            f"got {detectable_shift!r}.")
    if "mode" in detectable_shift:
        raise ValueError(
            "detectable_shift no longer accepts 'mode'; use 'indifference_zone_quantile'.")
    if "baseline_credibility" in detectable_shift:
        raise ValueError(
            "detectable_shift uses 'indifference_zone_quantile', not 'baseline_credibility'.")
    unknown = set(detectable_shift) - SHIFT_SPECIFICATION_KEYS
    if unknown:
        raise ValueError(f"Unknown detectable_shift keys: {sorted(unknown)}.")
    if "indifference_zone_quantile" not in detectable_shift:
        raise ValueError(
            "detectable_shift is missing required key 'indifference_zone_quantile'.")
    assert_indifference_zone_quantile(detectable_shift["indifference_zone_quantile"])
    return detectable_shift


def assert_indifference_zone_quantile(quantile):
    """Require a quantile strictly between 0.5 and 1."""
    if not 0.5 < quantile < 1.0:
        raise ValueError(
            "indifference_zone_quantile must be greater than 0.5 and less than 1.0; "
            f"got {quantile}.")
    return quantile


def log_likelihood_ratio_increments(reference_rate, detectable_shift, regime):
    """Return log-likelihood increments for flip and non-flip events."""
    if not 0.0 < reference_rate < 1.0:
        raise ValueError(
            f"Reference flip rate for regime {regime} must be in (0, 1); "
            f"got {reference_rate}.")
    shifted_rate = reference_rate + detectable_shift
    if shifted_rate >= 1.0:
        raise ValueError(
            f"Shifted flip rate for regime {regime} must be less than 1; got {shifted_rate} "
            f"(reference {reference_rate}, shift {detectable_shift}).")
    return (float(np.log(shifted_rate / reference_rate)),
            float(np.log((1.0 - shifted_rate) / (1.0 - reference_rate))))


def break_even_rate(reference_rate, detectable_shift, regime):
    """Return the flip rate with zero expected CUSUM increment."""
    flip, no_flip = log_likelihood_ratio_increments(reference_rate, detectable_shift, regime)
    return -no_flip / (flip - no_flip)


def noise_boundary_shift(reference_rate, posterior, credibility, regime):
    """Return the shift whose break-even rate matches the posterior quantile."""
    target = float(stats.beta(*posterior).ppf(credibility))
    if target <= reference_rate:
        raise ValueError(
            f"Posterior quantile for regime {regime} must exceed the reference rate; got "
            f"{target} at quantile {credibility}, reference {reference_rate}.")
    largest = (1.0 - reference_rate) * LARGEST_SHIFT_FRACTION
    if break_even_rate(reference_rate, largest, regime) < target:
        raise ValueError(
            f"Posterior quantile {target} for regime {regime} is unreachable from reference "
            f"{reference_rate} within the allowed shift range.")
    return float(optimize.brentq(
        lambda shift: break_even_rate(reference_rate, shift, regime) - target,
        SMALLEST_SHIFT, largest, xtol=SHIFT_ROOT_TOLERANCE, rtol=SHIFT_ROOT_TOLERANCE))


def positions_to_alarm(threshold, reference_rate, detectable_shift, share, regime):
    """Return expected stream positions to alarm at the shifted rate."""
    flip, no_flip = log_likelihood_ratio_increments(reference_rate, detectable_shift, regime)
    design_rate = reference_rate + detectable_shift
    per_item_drift = design_rate * flip + (1.0 - design_rate) * no_flip
    return threshold / (per_item_drift * share)


def raised_until_reachable(resolved, ceiling, out_of_reach):
    """Raise unreachable shifts to reachable values within the search tolerance."""
    keys = tuple(resolved)
    unreachable = out_of_reach(resolved, keys)
    if not unreachable:
        return dict(resolved)

    unreachable_at_the_ceiling = out_of_reach(
        {**resolved, **{key: ceiling[key] for key in unreachable}}, unreachable)
    if unreachable_at_the_ceiling:
        raise ThresholdOutOfReach(unreachable_at_the_ceiling[0])

    lower = {key: resolved[key] for key in unreachable}
    upper = {key: ceiling[key] for key in unreachable}
    while any(upper[key] - lower[key] > REACHABILITY_TOLERANCE for key in unreachable):
        trial = {**resolved,
                 **{key: 0.5 * (lower[key] + upper[key]) for key in unreachable}}
        still_unreachable = out_of_reach(trial, unreachable)
        for key in unreachable:
            if key in still_unreachable:
                lower[key] = trial[key]
            else:
                upper[key] = trial[key]
    return {**resolved, **upper}
