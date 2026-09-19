"""Monitor belief-threshold events with a CUSUM for each verdict regime and boundary."""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy import stats

from rcv.cusum import (
    JEFFREYS_PRIOR_WEIGHT,
    LARGEST_SHIFT_FRACTION,
    MINIMUM_STREAMS_IN_THE_TAIL,
    ThresholdOutOfReach,
    assert_shift_specification,
    at_reference_replay_seed,
    break_even_rate,
    log_likelihood_ratio_increments,
    noise_boundary_shift,
    positions_to_alarm,
    raised_until_reachable,
)
from rcv.guards import REGIMES, assert_regimes

JOINT = "joint"
BUDGET_ALLOCATIONS = (JOINT,)

CENTRE_VIOLATION_MINIMUM = 1


def violating_centre_fraction(peaks, threshold, centre_size, minimum_alarms):
    """Return the fraction of centres with at least minimum_alarms alarming streams."""
    alarming = np.logical_or.reduce([peaks[regime][boundary] >= threshold[regime][boundary]
                                     for regime in REGIMES for boundary in peaks[regime]])
    if alarming.size % centre_size:
        raise ValueError(f"{alarming.size} replayed streams do not divide into whole centres of "
                         f"{centre_size}.")
    if minimum_alarms > centre_size:
        raise ValueError(f"minimum_alarms {minimum_alarms} exceeds centre_size {centre_size}.")
    per_centre = alarming.reshape(-1, centre_size).sum(axis=1)
    return float(np.mean(per_centre >= minimum_alarms))


class BankAlarm(NamedTuple):
    """First threshold crossing; item_index is zero-based within the observed batch."""
    regime: int
    item_index: int
    boundary: float


@dataclass(frozen=True)
class BeliefBankCalibration:
    """Replay results; per-wire dictionaries are keyed by regime, then boundary.

    The calibration_level field records the achieved tuning fraction of violating centres.
    """

    reference: dict
    threshold: dict
    realized_alarm_rate_replayed: dict
    realized_alarm_rate_combined_replayed: float
    replay_stream_length: int
    n_streams: int
    seed: int
    realized_alarm_rate_at_reference: dict = None
    realized_alarm_rate_combined_at_reference: float = None
    monitored_length: int = None
    quiet_horizon_confidence: float = None
    detectable_shift: dict = None
    break_even: dict = None
    indifference_zone_quantile: float = None
    reference_posterior_quantile: dict = None
    resolved_detectable_shift: dict = None
    prefix_drift_checked: bool = None
    fitted_family_overlap: dict = None
    boundaries: tuple = ()
    budget_allocation: str = JOINT
    streams_per_centre: int = None
    wire_quantile_level: float = None
    calibration_level: float = None
    streams_in_the_tail: dict = field(default_factory=dict)


def assert_boundaries(boundaries):
    boundaries = tuple(float(boundary) for boundary in boundaries)
    if not boundaries:
        raise ValueError("At least one bank boundary is required.")
    if any(not 0.0 < boundary < 1.0 for boundary in boundaries):
        raise ValueError(f"Bank boundary outside (0, 1): {boundaries}.")
    if any(later <= earlier for earlier, later in zip(boundaries, boundaries[1:])):
        raise ValueError(f"Bank boundaries {boundaries} are not strictly ascending.")
    return boundaries


def assert_streams_per_centre(streams_per_centre):
    if not isinstance(streams_per_centre, int) or isinstance(streams_per_centre, bool):
        raise ValueError(f"streams_per_centre must be an integer; got {streams_per_centre!r}.")
    if streams_per_centre < 1:
        raise ValueError(f"streams_per_centre must be at least 1; got {streams_per_centre}.")
    return streams_per_centre


def assert_budget_allocation(budget_allocation):
    if budget_allocation == "bonferroni":
        raise ValueError("budget allocation 'bonferroni' is retired; use 'joint'.")
    if budget_allocation not in BUDGET_ALLOCATIONS:
        raise ValueError(f"budget allocation {budget_allocation!r} is not one of "
                         f"{BUDGET_ALLOCATIONS}.")
    return budget_allocation


def as_belief(probability):
    """Return a one-dimensional float64 array of probabilities in [0, 1]."""
    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 1:
        raise ValueError("Probability of agreement must be one-dimensional: one value per item.")
    if np.isnan(probability).any():
        raise ValueError("Probability of agreement contains NaN.")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("Probability of agreement outside [0, 1].")
    return probability


def as_regime_column(verdict):
    verdict = np.asarray(verdict)
    if verdict.dtype.kind == "f" and np.isnan(verdict).any():
        raise ValueError("Verdict contains NaN.")
    assert_regimes(verdict, "monitor routing")
    return verdict.astype(np.int64)


def as_observations(probability, verdict, what):
    probability, verdict = as_belief(probability), as_regime_column(verdict)
    if len(probability) != len(verdict):
        raise ValueError(f"{what} carries {len(probability)} beliefs against {len(verdict)} "
                         f"verdicts; lengths must match.")
    return probability, verdict


def events_of(probability, boundaries):
    """Return a (K, n) boolean array for n beliefs strictly below K boundaries."""
    probability = as_belief(probability)
    return probability[None, :] < np.asarray(boundaries, dtype=np.float64)[:, None]


def bin_counts_of(probability, verdict, boundaries, regime):
    """Count a regime's beliefs in K+1 bins; boundary ties enter the upper bin."""
    in_regime = probability[verdict == regime]
    edges = np.asarray(boundaries, dtype=np.float64)
    return np.bincount(np.searchsorted(edges, in_regime, side="right"), minlength=len(edges) + 1)


def wire_peak_accumulation(event, in_regime, reference_rate, detectable_shift, regime):
    """Return the peak zero-floored CUSUM, updating only for items in the regime."""
    on_event, off_event = log_likelihood_ratio_increments(reference_rate, detectable_shift,
                                                          regime)
    increment = np.where(in_regime, np.where(event, on_event, off_event), 0.0)
    unfloored = np.concatenate(([0.0], np.cumsum(increment)))
    return float(np.max(unfloored - np.minimum.accumulate(unfloored)))


class BeliefBankMonitor:
    """Monitor agreement probabilities below each boundary, separately by verdict.

    Inputs are aligned one-dimensional arrays; verdicts use 0 for safe and
    1 for unsafe. A wire is a (regime, boundary) pair. The calibration_level
    parameter bounds the tuning fraction of posterior centres with at least one
    alarming stream; streams_per_centre sets the replays per posterior draw.
    """

    def __init__(self, calibration_level, detectable_shift, boundaries, streams_per_centre=1,
                 budget_allocation=JOINT):
        if not 0.0 < calibration_level < 1.0:
            raise ValueError(f"Monitor calibration level must be in (0, 1); got {calibration_level}.")
        self.calibration_level = calibration_level
        self.detectable_shift = assert_shift_specification(detectable_shift)
        self.boundaries = assert_boundaries(boundaries)
        self.budget_allocation = assert_budget_allocation(budget_allocation)
        self.streams_per_centre = assert_streams_per_centre(streams_per_centre)
        self.calibration = None
        self.reset()

    @property
    def wires(self):
        return [(regime, boundary) for regime in REGIMES for boundary in self.boundaries]

    @property
    def accumulation(self):
        return {regime: dict(events) for regime, events in self._accumulated.items()}

    @property
    def items_observed(self):
        return self._items_observed

    def reset(self):
        """Clear accumulations and the observation count; retain calibration."""
        self._accumulated = {regime: {boundary: 0.0 for boundary in self.boundaries}
                             for regime in REGIMES}
        self._items_observed = 0

    def calibrate(self, drift_free_probability, drift_free_verdict, stream_length, n_streams,
                  seed, prefix_drift_checked=None,
                  monitored_length=None, fitted_family_overlap=None):
        """Set and return replay calibration; reset observation state on success.

        Supply drift-free reference data covering both regimes. Each wire's
        reference event rate must be strictly between 0 and 1. Replay lengths
        count items across both regimes; n_streams must form whole centres.
        Drift and fitting-overlap metadata are recorded as supplied.
        """
        if stream_length < 1:
            raise ValueError(f"Replay stream_length must be at least 1; got {stream_length}.")
        self._assert_the_tail_can_be_read(n_streams)
        probability, verdict = as_observations(drift_free_probability, drift_free_verdict,
                                               "the drift-free material")

        counts = self._bin_counts(probability, verdict)
        reference = self._reference_from(counts)
        posterior = self._dirichlet_posterior(counts)
        resolution = self._resolved_shift(probability, verdict, reference, posterior,
                                          stream_length, n_streams, seed)
        shift = resolution["detectable_shift"]

        event = events_of(probability, self.boundaries)
        replay = np.random.default_rng(seed)

        def next_replayed_streams():
            return self._replayed_peaks(event, verdict, reference, shift, stream_length,
                                        n_streams, replay, posterior)

        tuning_peaks = next_replayed_streams()
        streams_in_the_tail, tuning_union = self._streams_in_the_tail(tuning_peaks, n_streams)
        threshold = self._threshold_from(tuning_peaks, streams_in_the_tail)
        measurement_peaks = next_replayed_streams()
        realized, combined = self._alarm_rate_of(measurement_peaks, threshold)
        at_reference_peaks = self._replayed_peaks(
            event, verdict, reference, shift, stream_length, n_streams,
            np.random.default_rng(at_reference_replay_seed(seed)), posterior,
            at_the_reference=True)
        at_reference, at_reference_combined = self._alarm_rate_of(at_reference_peaks, threshold)

        self.calibration = BeliefBankCalibration(
            reference=reference, threshold=threshold,
            realized_alarm_rate_replayed=realized,
            realized_alarm_rate_combined_replayed=combined,
            realized_alarm_rate_at_reference=at_reference,
            realized_alarm_rate_combined_at_reference=at_reference_combined,
            replay_stream_length=stream_length, monitored_length=monitored_length,
            n_streams=n_streams, seed=seed,
            quiet_horizon_confidence=1.0 - self.calibration_level,
            detectable_shift=shift,
            break_even=self._per_wire(lambda regime, boundary: break_even_rate(
                reference[regime][boundary], shift[regime][boundary], regime)),
            indifference_zone_quantile=resolution["indifference_zone_quantile"],
            reference_posterior_quantile=resolution["reference_posterior_quantile"],
            resolved_detectable_shift=resolution["resolved_detectable_shift"],
            prefix_drift_checked=prefix_drift_checked,
            fitted_family_overlap=fitted_family_overlap,
            boundaries=self.boundaries, budget_allocation=self.budget_allocation,
            streams_per_centre=self.streams_per_centre,
            wire_quantile_level=self._wire_quantile_level(streams_in_the_tail, n_streams),
            calibration_level=tuning_union,
            streams_in_the_tail=streams_in_the_tail)
        self.reset()
        return self.calibration

    def observe(self, probability, verdict):
        """Update state until the first alarm; return BankAlarm or None.

        State persists across calls. An alarm stops processing immediately,
        including later boundaries for that item and the rest of the batch.
        """
        if self.calibration is None:
            raise ValueError("Monitor is not calibrated; call calibrate before observe.")
        probability, verdict = as_observations(probability, verdict, "the traffic")
        reference, shift = self.calibration.reference, self.calibration.detectable_shift
        threshold = self.calibration.threshold
        increments = self._per_wire(lambda regime, boundary: log_likelihood_ratio_increments(
            reference[regime][boundary], shift[regime][boundary], regime))
        boundaries = np.asarray(self.boundaries, dtype=np.float64)
        for item_index, (belief, regime) in enumerate(zip(probability.tolist(),
                                                          verdict.tolist())):
            fired = belief < boundaries
            self._items_observed += 1
            for position, boundary in enumerate(self.boundaries):
                on_event, off_event = increments[regime][boundary]
                increment = on_event if fired[position] else off_event
                self._accumulated[regime][boundary] = max(
                    0.0, self._accumulated[regime][boundary] + increment)
                if self._accumulated[regime][boundary] >= threshold[regime][boundary]:
                    return BankAlarm(regime=regime, item_index=item_index, boundary=boundary)
        return None


    def _per_wire(self, of_wire):
        return {regime: {boundary: of_wire(regime, boundary) for boundary in self.boundaries}
                for regime in REGIMES}

    def _bin_counts(self, probability, verdict):
        counts = {}
        for regime in REGIMES:
            if not np.any(verdict == regime):
                raise ValueError(f"Reference data has no rows in regime {regime}.")
            counts[regime] = bin_counts_of(probability, verdict, self.boundaries, regime)
        return counts

    def _reference_from(self, counts):
        return {regime: {boundary: float(rate) for boundary, rate in zip(
                    self.boundaries, np.cumsum(counts[regime])[:-1] / counts[regime].sum())}
                for regime in REGIMES}

    @staticmethod
    def _dirichlet_posterior(counts):
        """Return Dirichlet parameters: bin counts plus the Jeffreys prior of 1/2."""
        return {regime: counts[regime].astype(np.float64) + JEFFREYS_PRIOR_WEIGHT
                for regime in REGIMES}

    def _event_posterior(self, posterior, regime, position):
        """Return Beta parameters for one wire's marginal event rate."""
        alpha = posterior[regime]
        return float(alpha[:position + 1].sum()), float(alpha[position + 1:].sum())

    def _resolved_shift(self, probability, verdict, reference, posterior, stream_length,
                        n_streams, seed):
        """Resolve posterior-quantile shifts, then raise them to meet the drift-based horizon estimate."""
        credibility = float(self.detectable_shift["indifference_zone_quantile"])
        marginal = {regime: {boundary: self._event_posterior(posterior, regime, position)
                             for position, boundary in enumerate(self.boundaries)}
                    for regime in REGIMES}
        quantile = self._per_wire(lambda regime, boundary: float(
            stats.beta(*marginal[regime][boundary]).ppf(credibility)))
        resolved = self._per_wire(lambda regime, boundary: noise_boundary_shift(
            reference[regime][boundary], marginal[regime][boundary], credibility, regime))
        share = {regime: float(np.mean(verdict == regime)) for regime in REGIMES}
        event = events_of(probability, self.boundaries)

        def out_of_reach(shift, wires):
            threshold = self._thresholds_at(event, verdict, reference, shift, stream_length,
                                            n_streams, seed, posterior)
            return [(regime, boundary) for regime, boundary in wires
                    if positions_to_alarm(threshold[regime][boundary], reference[regime][boundary],
                                          shift[regime][boundary], share[regime],
                                          regime) > stream_length]

        return {"detectable_shift": self._raised_until_reachable(resolved, reference, out_of_reach,
                                                                 stream_length),
                "indifference_zone_quantile": credibility,
                "reference_posterior_quantile": quantile,
                "resolved_detectable_shift": resolved}

    def _raised_until_reachable(self, resolved, reference, out_of_reach, stream_length):
        flattened = {wire: resolved[wire[0]][wire[1]] for wire in self.wires}
        ceiling = {wire: (1.0 - reference[wire[0]][wire[1]]) * LARGEST_SHIFT_FRACTION
                   for wire in self.wires}

        def out_of_reach_flat(shift, wires):
            return out_of_reach(self._unflattened(shift), wires)

        try:
            raised = raised_until_reachable(flattened, ceiling, out_of_reach_flat)
        except ThresholdOutOfReach as unreachable:
            regime, boundary = unreachable.key
            raise ValueError(
                f"Regime {regime}, boundary {boundary}: estimated positions to alarm exceed "
                f"the {stream_length}-position horizon at the shift ceiling.") from None
        return self._unflattened(raised)

    def _unflattened(self, by_wire):
        return self._per_wire(lambda regime, boundary: by_wire[(regime, boundary)])

    def _thresholds_at(self, event, verdict, reference, shift, stream_length, n_streams, seed,
                       posterior):
        peaks = self._replayed_peaks(event, verdict, reference, shift, stream_length, n_streams,
                                     np.random.default_rng(seed), posterior)
        streams_in_the_tail, _ = self._streams_in_the_tail(peaks, n_streams)
        return self._threshold_from(peaks, streams_in_the_tail)


    def _replayed_peaks(self, event, verdict, reference, shift, stream_length, n_streams, replay,
                        posterior, at_the_reference=False):
        """Return per-wire peaks from replays with resampled verdicts.

        Event rates are drawn from the Dirichlet posterior per centre, with
        shared uniforms preserving nested events. With at_the_reference=True,
        observed event/verdict pairs are resampled instead.
        """
        peaks = {regime: {boundary: np.empty(n_streams) for boundary in self.boundaries}
                 for regime in REGIMES}
        for stream in range(n_streams):
            drawn = replay.integers(0, event.shape[1], size=stream_length)
            drawn_verdict = verdict[drawn]
            if not at_the_reference:
                if stream % self.streams_per_centre == 0:
                    cumulative = np.stack([self._drawn_cumulative_rates(posterior[regime], replay)
                                           for regime in REGIMES])
                uniform = replay.random(stream_length)
                rate = cumulative[np.searchsorted(REGIMES, drawn_verdict)]
                drawn_event = uniform[None, :] < rate.T
            else:
                drawn_event = event[:, drawn]
            for regime in REGIMES:
                in_regime = drawn_verdict == regime
                for position, boundary in enumerate(self.boundaries):
                    peaks[regime][boundary][stream] = wire_peak_accumulation(
                        drawn_event[position], in_regime, reference[regime][boundary],
                        shift[regime][boundary], regime)
        return peaks

    def _drawn_cumulative_rates(self, alpha, replay):
        """Draw cumulative event rates via Dirichlet stick breaking."""
        cumulative = np.empty(len(self.boundaries))
        remaining_mass, stick, total = float(alpha.sum()), 1.0, 0.0
        for position in range(len(self.boundaries)):
            share = float(replay.beta(alpha[position], remaining_mass - alpha[position])) * stick
            total += share
            cumulative[position] = total
            stick -= share
            remaining_mass -= alpha[position]
        return cumulative


    def _assert_the_tail_can_be_read(self, n_streams):
        if int(np.floor(self.calibration_level * n_streams)) < 1:
            raise ValueError(f"a calibration level of {self.calibration_level} over {n_streams} "
                             f"replayed streams leaves no stream in the tail.")

    def _streams_in_the_tail(self, peaks, n_streams):
        return self._joint_tail(peaks, n_streams)

    def _joint_tail(self, peaks, n_streams):
        """Return the largest common tail count meeting calibration_level and its centre fraction."""
        sorted_peaks = {wire: np.sort(peaks[wire[0]][wire[1]]) for wire in self.wires}

        def union_at(count):
            threshold = {wire: sorted_peaks[wire][-count] for wire in self.wires}
            return violating_centre_fraction(
                peaks, self._unflattened(threshold), self.streams_per_centre,
                CENTRE_VIOLATION_MINIMUM)

        largest = int(np.floor(self.calibration_level * n_streams))
        if union_at(1) > self.calibration_level:
            raise ValueError(
                f"At tail count 1, violating-centre fraction {union_at(1)} exceeds "
                f"calibration level {self.calibration_level} "
                f"({len(self.boundaries)} boundaries, {n_streams} streams).")
        low, high = 1, largest
        while low < high:
            middle = (low + high + 1) // 2
            if union_at(middle) <= self.calibration_level:
                low = middle
            else:
                high = middle - 1
        if low < MINIMUM_STREAMS_IN_THE_TAIL:
            raise ValueError(
                f"Joint calibration tail count {low} is below the required minimum "
                f"{MINIMUM_STREAMS_IN_THE_TAIL} per wire.")
        return self._per_wire(lambda regime, boundary: low), union_at(low)

    def _union_rate_at(self, peaks, streams_in_the_tail):
        threshold = self._threshold_from(peaks, streams_in_the_tail)
        return float(np.mean(np.logical_or.reduce(
            [peaks[regime][boundary] >= threshold[regime][boundary]
             for regime, boundary in self.wires])))

    def _wire_quantile_level(self, streams_in_the_tail, n_streams):
        levels = {streams_in_the_tail[regime][boundary] for regime, boundary in self.wires}
        return float(levels.pop() / n_streams) if len(levels) == 1 else None

    def _threshold_from(self, peaks, streams_in_the_tail):
        """Select per-wire order statistics; ties can put additional streams at the threshold."""
        threshold = {}
        for regime in REGIMES:
            threshold[regime] = {}
            for boundary in self.boundaries:
                in_the_tail = np.sort(peaks[regime][boundary])[
                    -streams_in_the_tail[regime][boundary]]
                if in_the_tail <= 0.0:
                    raise ValueError(
                        f"Regime {regime}, boundary {boundary}: selected replay threshold "
                        f"must be positive.")
                threshold[regime][boundary] = float(in_the_tail)
        return threshold

    def _alarm_rate_of(self, peaks, threshold):
        """Return per-wire and any-wire alarm fractions over the supplied streams."""
        alarmed = self._per_wire(
            lambda regime, boundary: peaks[regime][boundary] >= threshold[regime][boundary])
        realized = self._per_wire(
            lambda regime, boundary: float(np.mean(alarmed[regime][boundary])))
        every_wire = [alarmed[regime][boundary] for regime, boundary in self.wires]
        return realized, float(np.mean(np.logical_or.reduce(every_wire)))
