"""Group-blocked, optionally stratified splits and family-level cross-fitting."""

import numpy as np

MIXED_STRATUM = -1

BIT_GENERATOR = np.random.PCG64


def draw_group_blocked_split(family, evaluation_fraction, calibration_fraction, seed,
                             strata=None):
    """Allocate whole families to evaluation first, then calibration and training."""
    for name, fraction in (("evaluation_fraction", evaluation_fraction),
                           ("calibration_fraction", calibration_fraction)):
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"{name} must be in (0, 1); got {fraction!r}.")

    families = np.unique(family, equal_nan=True, sorted=True)
    rng = np.random.Generator(BIT_GENERATOR(seed))
    allocation = {"training": [], "calibration": [], "evaluation": []}
    for group_of_families in _families_by_stratum(family, families, strata):
        shuffled = rng.permutation(group_of_families)
        n_evaluation = round(len(shuffled) * evaluation_fraction)
        allocation["evaluation"].append(shuffled[:n_evaluation])
        remainder = shuffled[n_evaluation:]
        n_calibration = round(len(remainder) * calibration_fraction)
        allocation["calibration"].append(remainder[:n_calibration])
        allocation["training"].append(remainder[n_calibration:])

    split_families = {name: np.concatenate(chosen) for name, chosen in allocation.items()}
    for name, chosen in split_families.items():
        if len(chosen) == 0:
            raise ValueError(
                f"{name.capitalize()} split has no families after allocation "
                f"(evaluation_fraction={evaluation_fraction}, "
                f"calibration_fraction={calibration_fraction}, "
                f"families={len(families)}).")
    return {name: np.isin(family, chosen) for name, chosen in split_families.items()}


def _families_by_stratum(family, families, strata):
    """Group families deterministically by sole stratum, keeping mixed families together."""
    if strata is None:
        return [families]
    strata = np.asarray(strata)
    if len(strata) != len(family):
        raise ValueError(f"strata and family must have equal lengths; got {len(strata)} and "
                         f"{len(family)}.")
    stratum_of_family = {}
    for one_family in families:
        family_strata = np.unique(strata[family == one_family], equal_nan=True, sorted=True)
        stratum_of_family[one_family] = (int(family_strata[0]) if len(family_strata) == 1
                                         else MIXED_STRATUM)
    order = sorted(set(stratum_of_family.values()))
    return [np.array([f for f in families if stratum_of_family[f] == stratum])
            for stratum in order]


def cross_fit_fold_of(family, folds, rng):
    """Assign each whole family to a cross-fitting fold."""
    families = np.unique(family, equal_nan=True, sorted=True)
    shuffled = rng.permutation(families)
    fold_of_family = {one_family: position % folds
                      for position, one_family in enumerate(shuffled)}
    return np.array([fold_of_family[one_family] for one_family in family])
