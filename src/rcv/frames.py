"""Load frame archives and create estimator slices."""

import hashlib
from typing import NamedTuple

import numpy as np


class Slice(NamedTuple):
    """Estimator inputs: representation, verdict, and agreement arrays."""
    representation: np.ndarray
    verdict: np.ndarray
    agreement: np.ndarray


def sha256_of(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_frame(path):
    with np.load(path, allow_pickle=False) as stored:
        frame = {name: stored[name] for name in stored.files}
    if "is_stream" in frame and frame["is_stream"].dtype != np.bool_:
        raise ValueError(f"is_stream must have boolean dtype; got {frame['is_stream'].dtype}.")
    return frame


def rows(frame, mask):
    return {name: values[mask] for name, values in frame.items()}


def concatenated(first, second):
    return {name: np.concatenate([first[name], second[name]]) for name in first}


def agreement_of(frame, labels):
    """Return agreement between verdict and a named label column."""
    if labels not in frame:
        raise ValueError(f"Frame has no label column {labels!r}; available columns: "
                         f"{sorted(frame)}.")
    return (frame["verdict"] == frame[labels]).astype(int)


def slice_of(frame, labels):
    """Return estimator inputs using a named label column."""
    return Slice(frame["representation"], frame["verdict"], agreement_of(frame, labels))


def fit_on_slices(estimator, fit_slice, calibration_slice, labels):
    """Fit an estimator on fitting and calibration frame slices."""
    return estimator.fit(fitting_slice=slice_of(fit_slice, labels),
                         calibration_slice=slice_of(calibration_slice, labels))
