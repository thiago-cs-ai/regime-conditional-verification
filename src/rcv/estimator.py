"""Correctness estimation with pooled or per-verdict probes and calibrations."""

import numpy as np

from rcv.calibrations import build_calibration
from rcv.guards import (
    POOLED,
    REGIMES,
    assert_all_four_cases,
    assert_regimes,
    assert_scores_rankable,
)
from rcv.probes import build_probe

EVERY_ROW = slice(None)


class Estimator:
    """Each route flag enables per-verdict models for its stage; False pools rows.

    Pass probe and calibration settings as dictionaries. Inputs are aligned
    NumPy arrays: representation (n, d), verdict (n,), and agreement (n,).
    Verdicts use 0 for safe and 1 for unsafe; agreement is 1 when the
    verdict matches the target label, 0 otherwise.
    """

    def __init__(self, probe, calibration, route_probe, route_calibration, random_state=None):
        self._probe_spec = probe
        self._calibration_spec = calibration
        self._route_probe = route_probe
        self._route_calibration = route_calibration
        self._random_state = random_state
        self._probe_by_group = {}
        self._calibration_by_group = {}

    def fit(self, fitting_slice, calibration_slice):
        """Fit probes and calibrations; return self.

        Supply disjoint rcv.frames.Slice inputs, each containing all four
        verdict/agreement combinations. When configured, probe selection also
        uses the calibration slice.
        """
        assert_all_four_cases(fitting_slice, "fitting")
        assert_all_four_cases(calibration_slice, "calibration")
        self._fit_probes(fitting_slice, calibration_slice)
        self._fit_calibrations(calibration_slice)
        return self

    def _fit_probes(self, fitting_slice, calibration_slice):
        held_out_rows = dict(self._row_groups(self._route_probe, calibration_slice.verdict))
        for group, rows in self._row_groups(self._route_probe, fitting_slice.verdict):
            probe = build_probe(self._probe_spec, self._random_state)
            if probe.needs_validation:
                fold = held_out_rows[group]
                probe.fit(fitting_slice.representation[rows], fitting_slice.agreement[rows],
                          calibration_slice.representation[fold],
                          calibration_slice.agreement[fold])
            else:
                probe.fit(fitting_slice.representation[rows], fitting_slice.agreement[rows])
            self._probe_by_group[group] = probe

    def probe_selection(self):
        """Return selected parameter records, each with group, parameter, and value; otherwise None."""
        selected = [{"group": group, **row}
                    for group in sorted(self._probe_by_group)
                    for row in (self._probe_by_group[group].selection or ())]
        return selected or None

    def _fit_calibrations(self, calibration_slice):
        held_out_scores = self.probe_scores(calibration_slice.representation,
                                            calibration_slice.verdict)
        for group, rows in self._row_groups(self._route_calibration, calibration_slice.verdict):
            self._calibration_by_group[group] = build_calibration(self._calibration_spec).fit(
                held_out_scores[rows], calibration_slice.agreement[rows])

    def probe_scores(self, representation, verdict):
        """Return row-aligned float64 logits of agreement."""
        scores = np.empty(len(verdict), dtype=np.float64)
        for group, rows in self._row_groups(self._route_probe, verdict):
            probe = self._probe_by_group[group]
            raw_scores = probe.logit_scores(representation[rows])
            scores[rows] = assert_scores_rankable(
                raw_scores, f"probe[{group}]", inputs=representation[rows],
                check_pinning=probe.scores_are_continuous)
        return scores

    def probability_from_scores(self, scores, verdict):
        """Calibrate row-aligned logits to probabilities of agreement."""
        probabilities = np.empty(len(verdict), dtype=np.float64)
        for group, rows in self._row_groups(self._route_calibration, verdict):
            probabilities[rows] = self._calibration_by_group[group].probability(scores[rows])
        return probabilities

    def probability_of_agreement(self, representation, verdict):
        """Return the calibrated probability that each verdict matches the target label."""
        return self.probability_from_scores(self.probe_scores(representation, verdict), verdict)

    @staticmethod
    def _row_groups(routed, verdict):
        if not routed:
            return [(POOLED, EVERY_ROW)]
        assert_regimes(verdict, "estimator routing")
        candidate_groups = [(regime, verdict == regime) for regime in REGIMES]
        return [(regime, rows) for regime, rows in candidate_groups if rows.any()]
