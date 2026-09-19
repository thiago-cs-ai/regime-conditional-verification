import numpy as np
import pytest
import yaml

from fixture_frame import FIXTURE_REFERENCE_WINDOW, build_fixture_frame
from rcv.frames import rows
from rcv.study import CENSUS_BASELINE_COMPONENT


class TestAuditDraw:
    def test_the_draw_is_uniform_within_the_post_alarm_window(self):
        from rcv.loop import draw_audit_indices

        drawn = draw_audit_indices(post_alarm_start=100, traffic_length=400,
                                   sampling_window=200, budget=50,
                                   rng=np.random.default_rng(0))
        assert len(drawn) == 50
        assert len(np.unique(drawn)) == 50
        assert drawn.min() >= 100
        assert drawn.max() < 300

        coverage = np.concatenate([
            draw_audit_indices(100, 400, 200, 50, np.random.default_rng(s))
            for s in range(200)])
        covered = np.unique(coverage)
        assert covered.min() < 110 and covered.max() >= 290, "draws concentrate instead of covering"

    def test_the_draw_is_seeded_and_deterministic(self):
        from rcv.loop import draw_audit_indices

        first = draw_audit_indices(0, 500, 300, 80, np.random.default_rng(7))
        again = draw_audit_indices(0, 500, 300, 80, np.random.default_rng(7))
        moved = draw_audit_indices(0, 500, 300, 80, np.random.default_rng(8))
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, moved)

    def test_a_short_tail_yields_what_exists(self):
        from rcv.loop import draw_audit_indices

        drawn = draw_audit_indices(post_alarm_start=380, traffic_length=400,
                                   sampling_window=200, budget=50,
                                   rng=np.random.default_rng(0))
        assert len(drawn) == 20
        assert drawn.min() >= 380 and drawn.max() < 400


def fixture_study_config(frame_path, study_name):
    return {
        "study": study_name,
        "seeds": [0],
        "frame": {"path": str(frame_path), "name": "synthetic-fixture",
                  "fitting_labels": "oracle"},
        "split": {"evaluation_fraction": 0.3, "calibration_fraction": 0.25},
        "estimator": {"probe": {"family": "linear"}, "calibration": {"method": "platt"},
                      "route_probe": True, "route_calibration": True},
        "flip": {"threshold": 0.5},
        "monitor": {"quiet_horizon_confidence": 0.95,
                    "detectable_shift": {"indifference_zone_quantile": 0.99},
                    "event_bank": {"boundaries": [0.5], "budget_allocation": "joint"},
                    "calibration": {"stream_length": 120, "n_streams": 2000, "streams_per_centre": 1}},
        "loop": {"audit_sampling_window": 120, "audit_budget": 60,
                 "acceptance": {"recall_floor": 0.5, "false_positive_tolerance": 0.25},
                 "max_enlarging_retries": 2, "cross_fit_folds": 4,
                 "post_repair_reference_window": FIXTURE_REFERENCE_WINDOW},
    }


@pytest.fixture(scope="module")
def deployment_result(tmp_path_factory):
    from rcv.study import run_study

    tmp_path = tmp_path_factory.mktemp("shape")
    frame = build_fixture_frame()
    frame_path = tmp_path / "frame.npz"
    np.savez(frame_path, **frame)
    config = fixture_study_config(frame_path, "audit-shape")
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return run_study(config_path)


class TestLabelPurchaseDecoupledFromObservation:
    def test_the_first_audit_spends_the_budget_and_only_the_budget(self, deployment_result):
        first_repair = deployment_result.change_log[0]
        assert first_repair.oracle_labels_spent % 60 == 0
        assert first_repair.oracle_labels_spent >= 60

    def test_the_loop_events_name_the_draw(self, deployment_result):
        audit_details = [event.detail for event in deployment_result.loop.events
                         if event.kind == "audit"]
        assert audit_details, "an alarm must produce an audit"
        assert "sampled uniformly" in audit_details[0]


class TestObserveOnly:
    def test_an_observe_only_watch_records_the_first_alarm_and_stops(self, tmp_path):
        import yaml

        from fixture_frame import build_fixture_frame
        from rcv.study import run_study

        frame = build_fixture_frame()
        frame_path = tmp_path / "frame.npz"
        np.savez(frame_path, **frame)
        config = fixture_study_config(frame_path, "observe-only-fixture")
        config["loop"]["observe_only"] = True
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = run_study(config_path)

        kinds = [event.kind for event in result.loop.events]
        assert kinds.count("alarm") == 1
        assert "audit" not in kinds and "repair_accepted" not in kinds
        assert result.change_log == []
        assert result.total_oracle_labels_spent == result.readout.oracle_labels_spent


class TestBaselineRelativeAcceptance:
    def test_the_criteria_arithmetic_on_a_hand_case(self):
        from rcv.loop import update_meets_relative_criteria

        assert update_meets_relative_criteria(
            update_recall=0.85, update_over_block=0.11,
            baseline_recall=0.80, baseline_over_block=0.10,
            recall_slack=0.02, over_block_inflation=0.02) is True
        assert update_meets_relative_criteria(
            update_recall=0.77, update_over_block=0.10,
            baseline_recall=0.80, baseline_over_block=0.10,
            recall_slack=0.02, over_block_inflation=0.02) is False
        assert update_meets_relative_criteria(
            update_recall=0.85, update_over_block=0.13,
            baseline_recall=0.80, baseline_over_block=0.10,
            recall_slack=0.02, over_block_inflation=0.02) is False

    def test_a_relative_acceptance_study_completes_the_cycle(self, tmp_path):
        import yaml

        from fixture_frame import build_fixture_frame
        from rcv.study import run_study

        frame = build_fixture_frame()
        frame_path = tmp_path / "frame.npz"
        np.savez(frame_path, **frame)
        config = fixture_study_config(frame_path, "relative-acceptance-fixture")
        config["loop"]["acceptance"] = {"form": "baseline_relative", "recall_slack": 0.15,
                                        "over_block_inflation": 0.10}
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = run_study(config_path)
        kinds = [event.kind for event in result.loop.events]
        assert "repair_accepted" in kinds
        assert result.change_log


class TestEscalationAsOutcome:
    def test_a_failed_acceptance_ends_the_watch_as_a_recorded_escalation(self, tmp_path):
        from rcv.study import run_study

        frame = build_fixture_frame()
        frame_path = tmp_path / "frame.npz"
        np.savez(frame_path, **frame)
        config = fixture_study_config(frame_path, "escalation-fixture")
        config["loop"]["acceptance"] = {"recall_floor": 1.0, "false_positive_tolerance": 0.0}
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = run_study(config_path)

        kinds = [event.kind for event in result.loop.events]
        assert kinds.count("escalation_demanded") == 1
        assert kinds[-1] == "escalation_demanded"
        assert "repair_accepted" not in kinds
        escalations = [entry for entry in result.change_log if entry.what == "escalation_demanded"]
        assert len(escalations) == 1
        assert escalations[0].oracle_labels_spent >= 60
        assert result.total_oracle_labels_spent >= result.readout.oracle_labels_spent + 60


class TestNoUnevaluatedAuditPurchase:
    def test_an_escalation_spends_only_what_was_evaluated(self, tmp_path):
        from rcv.study import run_study

        frame = build_fixture_frame()
        frame_path = tmp_path / "frame.npz"
        np.savez(frame_path, **frame)
        config = fixture_study_config(frame_path, "no-dead-spend-fixture")
        config["loop"]["acceptance"] = {"recall_floor": 1.0, "false_positive_tolerance": 0.0}
        config["loop"]["max_enlarging_retries"] = 0
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = run_study(config_path)

        escalation = next(entry for entry in result.change_log
                          if entry.what == "escalation_demanded")
        assert escalation.oracle_labels_spent == config["loop"]["audit_budget"]


class TestTheEpisodeModeAtStudyGrade:
    def episode_result(self, tmp_path, **loop_extras):
        import yaml

        from rcv.study import run_study

        frame = build_fixture_frame()
        frame_path = tmp_path / "frame.npz"
        np.savez(frame_path, **frame)
        config = fixture_study_config(frame_path, "episode-fixture")
        config["loop"]["mode"] = "episode"
        del config["loop"]["post_repair_reference_window"]
        config["loop"].update(loop_extras)
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))
        return run_study(config_path)

    def test_the_repair_resolves_the_unit_and_the_watch_ends(self, tmp_path):
        result = self.episode_result(tmp_path)

        kinds = [event.kind for event in result.loop.events]
        assert kinds.count("alarm") == 1
        assert kinds[-2:] == ["repair_accepted", "watch_ended"]
        assert "reference_rederived" not in kinds
        assert result.monitor.recalibrations == [], "an episode leaves the deployed calibration"

    def test_the_repair_is_billed_with_the_rates_the_gate_accepted_it_on(self, tmp_path):
        result = self.episode_result(tmp_path)

        repair = result.change_log[0]
        assert repair.what == "repair"
        assert repair.oracle_labels_spent >= 60
        assert 0.0 <= repair.gate_recall <= 1.0 and 0.0 <= repair.gate_over_block <= 1.0
        assert result.total_oracle_labels_spent == (result.readout.oracle_labels_spent
                                                    + repair.oracle_labels_spent)

    def test_every_acceptance_evaluation_is_on_the_record(self, tmp_path):
        result = self.episode_result(tmp_path)

        evaluations = [event for event in result.loop.events
                       if event.kind == "acceptance_evaluated"]
        assert evaluations, "an audit must be judged"
        assert evaluations[-1].gate_passed is True
        assert evaluations[-1].audit_labels >= 60
        assert all(event.gate_recall is not None for event in evaluations)

    def test_an_escalating_episode_is_terminal_exactly_as_under_deployment(self, tmp_path):
        result = self.episode_result(
            tmp_path, acceptance={"recall_floor": 1.0, "false_positive_tolerance": 0.0})

        kinds = [event.kind for event in result.loop.events]
        assert kinds[-1] == "escalation_demanded"
        assert "repair_accepted" not in kinds


FIXTURE_RAMP_CAP = 0.30


def ramped_fixture_frame():
    frame = build_fixture_frame()
    stream = frame["is_stream"]
    schedule = np.zeros(len(stream))
    schedule[stream] = np.linspace(0.0, FIXTURE_RAMP_CAP, int(stream.sum()), endpoint=False)
    frame["stream_lambda"] = schedule
    frame["is_drift_item"] = stream & (np.arange(len(stream)) % 2 == 0)
    return frame


class TestTheAlarmTimeDistributionReachesTheLoop:
    LADDER = 180
    GATE_LABELS = 40

    def run_with_doubles(self, tmp_path, composed, **loop_extras):
        import yaml

        from rcv import chaining
        from rcv.study import run_study

        frame = ramped_fixture_frame()
        frame_path = tmp_path / "frame.npz"
        np.savez(frame_path, **frame)
        config = fixture_study_config(frame_path, "post-alarm-fixture")
        config["loop"]["mode"] = "episode"
        del config["loop"]["post_repair_reference_window"]
        config["loop"]["post_alarm"] = {"composition": "stationary",
                                        "gate_labels": self.GATE_LABELS}
        config["loop"].update(loop_extras)
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        def post_alarm_pools(whole_frame):
            composed["pools_read_from"] = len(whole_frame["verdict"])
            stream = rows(whole_frame, whole_frame["is_stream"])
            return (rows(stream, ~stream["is_drift_item"]), rows(stream, stream["is_drift_item"]))

        def audit_and_gate_blocks(baseline, attack_pool, rate, audit_length, gate_length, rng,
                                  consumed_items=frozenset()):
            composed.update({"baseline": baseline, "rate": rate, "audit_length": audit_length,
                             "gate_length": gate_length, "consumed_items": consumed_items,
                             "draw": int(rng.integers(2 ** 32))})
            pool = baseline[CENSUS_BASELINE_COMPONENT][1]
            served = len(pool["verdict"])
            audit = rows(pool, np.arange(audit_length) % served)
            gate = rows(pool, (served - 1 - np.arange(gate_length)) % served)
            return audit, gate, {"rate": rate, "audit_length": audit_length,
                                 "gate_length": gate_length, "eligible_attack": 7,
                                 "eligible_base": 11, "audit_reused_rows": 2,
                                 "gate_reused_rows": 3, "audit_sha256": "a" * 64,
                                 "gate_sha256": "g" * 64}

        original = (getattr(chaining, "post_alarm_pools", None),
                    getattr(chaining, "audit_and_gate_blocks", None))
        chaining.post_alarm_pools = post_alarm_pools
        chaining.audit_and_gate_blocks = audit_and_gate_blocks
        try:
            return run_study(config_path)
        finally:
            for name, was in zip(("post_alarm_pools", "audit_and_gate_blocks"), original):
                if was is None:
                    delattr(chaining, name)
                else:
                    setattr(chaining, name, was)

    def test_both_blocks_are_composed_at_the_rate_the_alarm_fired_at(self, tmp_path):
        composed = {}
        result = self.run_with_doubles(tmp_path, composed)

        frame = ramped_fixture_frame()
        alarm = next(event for event in result.loop.events if event.kind == "alarm")
        assert composed["rate"] == frame["stream_lambda"][frame["is_stream"]][
            alarm.monitored_position]

    def test_the_audit_is_sized_by_the_ladder_and_the_gate_by_the_configuration(self, tmp_path):
        composed = {}
        self.run_with_doubles(tmp_path, composed)

        assert composed["audit_length"] == self.LADDER
        assert composed["gate_length"] == self.GATE_LABELS

    def test_the_pools_are_read_from_the_runs_own_frame(self, tmp_path):
        composed = {}
        self.run_with_doubles(tmp_path, composed)

        assert composed["pools_read_from"] == len(ramped_fixture_frame()["verdict"])
        assert composed["baseline"].keys() == {CENSUS_BASELINE_COMPONENT}

    def test_the_first_episodes_composer_is_told_nothing_is_consumed_yet(self, tmp_path):
        composed = {}
        self.run_with_doubles(tmp_path, composed)

        assert composed["consumed_items"] == frozenset()

    def test_the_blocks_seed_is_the_studys_and_the_alarms_position(self, tmp_path):
        from rcv.study import POST_ALARM_SEGMENT_SALT

        composed = {}
        result = self.run_with_doubles(tmp_path, composed)

        alarm = next(event for event in result.loop.events if event.kind == "alarm")
        expected = np.random.default_rng([result.seed, alarm.monitored_position,
                                          POST_ALARM_SEGMENT_SALT])
        assert composed["draw"] == int(expected.integers(2 ** 32)), (
            "the blocks are reproducible from the record: the study's seed and the alarm's "
            "own position are all a reader needs")

    def test_the_record_names_both_blocks_the_old_probe_and_the_whole_bill(self, tmp_path):
        composed = {}
        result = self.run_with_doubles(tmp_path, composed)

        entry = result.change_log[0]
        assert entry.post_alarm_rate == composed["rate"]
        assert entry.post_alarm_length == self.LADDER
        assert entry.post_alarm_gate_length == self.GATE_LABELS
        assert entry.gate_labels == self.GATE_LABELS
        assert entry.post_alarm_reused_rows == 5
        assert entry.old_probe_gate_recall is not None
        assert entry.old_probe_gate_over_block is not None
        accepted = next(event for event in result.loop.events
                        if event.kind == "acceptance_evaluated" and event.gate_passed)
        assert entry.oracle_labels_spent == accepted.audit_labels + self.GATE_LABELS

    def test_the_deciding_reading_is_the_gate_blocks_and_says_so(self, tmp_path):
        composed = {}
        result = self.run_with_doubles(tmp_path, composed)

        evaluation = next(event for event in result.loop.events
                          if event.kind == "acceptance_evaluated")
        assert f"on {self.GATE_LABELS} gate labels" in evaluation.detail
        assert evaluation.audit_labels == 60, "the material the candidate FITTED on"

    def test_the_old_probes_paired_reading_is_recorded_once(self, tmp_path):
        composed = {}
        result = self.run_with_doubles(tmp_path, composed)

        composed_events = [event for event in result.loop.events
                           if event.kind == "gate_block_composed"]
        assert len(composed_events) == 1
        assert composed_events[0].gate_recall == result.change_log[0].old_probe_gate_recall

    def test_a_study_naming_no_post_alarm_block_is_the_ramped_path_untouched(self, tmp_path):
        import yaml

        from rcv.study import run_study

        frame = build_fixture_frame()
        frame_path = tmp_path / "frame.npz"
        np.savez(frame_path, **frame)
        config = fixture_study_config(frame_path, "no-post-alarm-fixture")
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))
        result = run_study(config_path)

        repair = next(entry for entry in result.change_log if entry.what == "repair")
        assert repair.post_alarm_rate is None and repair.post_alarm_length is None
        assert repair.gate_labels is None and repair.old_probe_gate_recall is None
        assert "out-of-fold" in next(event.detail for event in result.loop.events
                                     if event.kind == "acceptance_evaluated")


class TestTheDataBoundedPolicyReachesTheLoop:
    REALIZED = 145
    GATE_LABELS = 40

    def run_with_doubles(self, tmp_path, composed, acceptance=None):
        import yaml

        from rcv import chaining
        from rcv.study import run_study

        frame = ramped_fixture_frame()
        frame_path = tmp_path / "frame.npz"
        np.savez(frame_path, **frame)
        config = fixture_study_config(frame_path, "data-bounded-fixture")
        config["loop"]["mode"] = "episode"
        del config["loop"]["post_repair_reference_window"]
        config["loop"]["post_alarm"] = {"composition": "stationary",
                                        "gate_labels": self.GATE_LABELS}
        config["loop"]["max_enlarging_retries"] = "exhaust-fresh-data"
        config["loop"]["acceptance"] = acceptance or {"recall_floor": 1.0,
                                                      "false_positive_tolerance": 0.0}
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(yaml.safe_dump(config))

        def post_alarm_pools(whole_frame):
            stream = rows(whole_frame, whole_frame["is_stream"])
            return (rows(stream, ~stream["is_drift_item"]), rows(stream, stream["is_drift_item"]))

        def audit_and_gate_blocks(baseline, attack_pool, rate, audit_length, gate_length, rng,
                                  consumed_items=frozenset()):
            composed["audit_length_asked_for"] = audit_length
            pool = baseline[CENSUS_BASELINE_COMPONENT][1]
            served = len(pool["verdict"])
            realized = self.REALIZED
            audit = rows(pool, np.arange(realized) % served)
            gate = rows(pool, (served - 1 - np.arange(gate_length)) % served)
            return audit, gate, {"rate": rate, "audit_length": realized,
                                 "gate_length": gate_length, "eligible_attack": 7,
                                 "eligible_base": 11, "audit_reused_rows": 0,
                                 "gate_reused_rows": 0, "audit_sha256": "a" * 64,
                                 "gate_sha256": "g" * 64}

        original = {name: getattr(chaining, name, None)
                    for name in ("post_alarm_pools", "audit_and_gate_blocks", "ALL_FRESH")}
        chaining.post_alarm_pools = post_alarm_pools
        chaining.audit_and_gate_blocks = audit_and_gate_blocks
        if original["ALL_FRESH"] is None:
            chaining.ALL_FRESH = "all-fresh"
        try:
            return run_study(config_path)
        finally:
            for name, was in original.items():
                if was is None:
                    delattr(chaining, name)
                else:
                    setattr(chaining, name, was)

    def test_the_composer_is_asked_for_every_fresh_label_rather_than_a_ladder(self, tmp_path):
        from rcv import chaining

        composed = {}
        self.run_with_doubles(tmp_path, composed)

        assert composed["audit_length_asked_for"] == getattr(chaining, "ALL_FRESH", "all-fresh")

    def test_the_attempts_climb_through_what_was_composed_and_end_on_all_of_it(self, tmp_path):
        composed = {}
        result = self.run_with_doubles(tmp_path, composed)

        curve = [event.audit_labels for event in result.loop.events
                 if event.kind == "acceptance_evaluated"]
        assert curve == [60, 120, self.REALIZED]

    def test_running_dry_is_recorded_as_its_own_terminal_with_the_whole_bill(self, tmp_path):
        composed = {}
        result = self.run_with_doubles(tmp_path, composed)

        kinds = [event.kind for event in result.loop.events]
        assert kinds[-1] == "fresh_data_exhausted"
        assert "escalation_demanded" not in kinds
        entry = result.change_log[0]
        assert entry.what == "fresh_data_exhausted"
        assert entry.oracle_labels_spent == self.REALIZED + self.GATE_LABELS
        assert entry.post_alarm_length == self.REALIZED, "the realized bound, not a ladder"
        assert entry.gate_labels == self.GATE_LABELS
        assert entry.old_probe_gate_recall is not None

    def test_a_repair_inside_the_data_bound_still_repairs(self, tmp_path):
        composed = {}
        result = self.run_with_doubles(tmp_path, composed,
                                       acceptance={"recall_floor": 0.0,
                                                   "false_positive_tolerance": 1.0})

        kinds = [event.kind for event in result.loop.events]
        assert "repair_accepted" in kinds and "fresh_data_exhausted" not in kinds
        assert result.change_log[0].oracle_labels_spent == 60 + self.GATE_LABELS
