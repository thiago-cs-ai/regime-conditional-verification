import numpy as np

REPRESENTATION_DIM = 16
ROWS_PER_FAMILY = 4
CALM_FAMILIES = 220
DRIFT_FAMILIES = 60
DRIFT_ERROR_RATE = 0.55
CALM_ERROR_RATE = 0.12
AGREEMENT_SIGNAL_STRENGTH = 4.0
FIXTURE_REFERENCE_WINDOW = 40


def build_fixture_frame(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    calm = _draw_segment(rng, n_families=CALM_FAMILIES, error_rate=CALM_ERROR_RATE,
                         family_offset=0, drifted=False)
    drift = _draw_segment(rng, n_families=DRIFT_FAMILIES, error_rate=DRIFT_ERROR_RATE,
                          family_offset=CALM_FAMILIES, drifted=True)
    return {key: np.concatenate([calm[key], drift[key]]) for key in calm}


def _draw_segment(rng, n_families: int, error_rate: float, family_offset: int,
                  drifted: bool) -> dict:
    n = n_families * ROWS_PER_FAMILY
    family = np.repeat(np.arange(family_offset, family_offset + n_families), ROWS_PER_FAMILY)
    item_id = np.arange(n) + family_offset * ROWS_PER_FAMILY
    verdict = rng.integers(0, 2, size=n)
    classifier_wrong = rng.random(n) < error_rate
    oracle = np.where(classifier_wrong, 1 - verdict, verdict)
    representation = _representation_encoding_agreement(rng, verdict, classifier_wrong)
    confidence_noise = rng.normal(0.0, 0.15, size=n)
    return {
        "family": family,
        "item_id": item_id,
        "verdict": verdict,
        "oracle": oracle,
        "representation": representation,
        "clf_score": np.clip(0.5 + 0.25 * np.where(classifier_wrong, -1.0, 1.0)
                             + confidence_noise, 0.0, 1.0),
        "is_stream": np.full(n, drifted),
    }


def _representation_encoding_agreement(rng, verdict, classifier_wrong):
    n = len(verdict)
    z = rng.normal(size=(n, REPRESENTATION_DIM))
    direction_by_regime = {0: _unit(rng.normal(size=REPRESENTATION_DIM)),
                          1: _unit(rng.normal(size=REPRESENTATION_DIM))}
    agreement_sign = np.where(classifier_wrong, -1.0, 1.0)
    for regime, direction in direction_by_regime.items():
        rows = verdict == regime
        z[rows] += AGREEMENT_SIGNAL_STRENGTH * agreement_sign[rows, None] * direction[None, :]
    return z


def _unit(v):
    return v / np.linalg.norm(v)


def split_for_fitting(frame, modulus: int = 4, calibration_residue: int = 3):
    from rcv.frames import rows

    calibration_rows = frame["family"] % modulus == calibration_residue
    return rows(frame, ~calibration_rows), rows(frame, calibration_rows)
