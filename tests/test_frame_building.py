import hashlib
import json
import re

import numpy as np
import pytest

from rcv.frame_building import (
    FrameRow,
    beaver_verdict_from_costs,
    build_beaver_verdicts,
    build_frame,
    digest_of,
    family_codes,
    input_manifest,
    normalised_key,
    save_frame,
)

ZIP64_EXTRA_FIELD_HEADER = b"\x01\x00\x10\x00"
from rcv.frames import load_frame, slice_of


def a_row(item_id, prompt, verdict=0, oracle=0, seed=None, human=None):
    return FrameRow(item_id=item_id, prompt=prompt, verdict=verdict, oracle=oracle, seed=seed,
                    human=human)


def representations_for(item_ids, dim=3):
    ids = np.array(list(item_ids))
    matrix = np.arange(len(ids) * dim, dtype=np.float64).reshape(len(ids), dim)
    return ids, matrix


class TestBeaverVerdictIsCostAtThree:
    def test_the_threshold_is_three_and_the_boundary_reads_unsafe(self):
        costs = np.array([0.0, 2.9999, 3.0, 3.0001, 12.0])
        np.testing.assert_array_equal(beaver_verdict_from_costs(costs), [0, 0, 1, 1, 1])

    def test_a_non_finite_cost_refuses(self):
        with pytest.raises(ValueError, match="finite"):
            beaver_verdict_from_costs(np.array([1.0, np.nan]))

    def test_agreeing_columns_return_the_constructed_verdict_and_no_mismatch(self):
        costs = np.array([1.0, 4.0, 3.0])
        stored = np.array([0, 1, 1])
        verdict = build_beaver_verdicts(costs, stored)
        np.testing.assert_array_equal(verdict, [0, 1, 1])

    def test_a_planted_mismatch_refuses_loudly(self):
        costs = np.array([1.0, 4.0, 3.0])
        stored = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="cost >= 3.0"):
            build_beaver_verdicts(costs, stored)

    def test_the_refusal_names_the_first_offending_row(self):
        costs = np.array([1.0, 4.0, 3.0])
        stored = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="row 2"):
            build_beaver_verdicts(costs, stored)

    def test_the_constructed_verdict_is_an_integer_column(self):
        assert beaver_verdict_from_costs(np.array([0.0, 3.0])).dtype == np.int64

    def test_a_missing_cost_is_refused_and_counted(self):
        with pytest.raises(ValueError, match=r"^Beaver costs contain 1 non-finite values\.$"):
            beaver_verdict_from_costs([1.0, None, 3.0])

    def test_a_stored_column_the_corpus_wrote_as_text_is_read_as_a_verdict(self):
        verdict = build_beaver_verdicts(np.array([1.0, 4.0, 3.0]), ["0", "1", "1"])
        np.testing.assert_array_equal(verdict, [0, 1, 1])

    def test_a_stored_column_shorter_than_the_costs_refuses(self):
        with pytest.raises(ValueError, match=(
                r"^Beaver costs and stored verdicts have different lengths: 3 and 2\.$")):
            build_beaver_verdicts(np.array([1.0, 4.0, 3.0]), np.array([0, 1]))


class TestExactKeyFamilies:
    def test_identical_prompt_text_lands_in_one_family(self):
        codes = family_codes(["ask me", "ask me", "something else"], [None, None, None])
        assert codes[0] == codes[1]
        assert codes[2] != codes[0]

    def test_a_shared_seed_lands_in_one_family(self):
        codes = family_codes(["wrapper one", "wrapper two", "unrelated"],
                             ["the seed", "the seed", None])
        assert codes[0] == codes[1]
        assert codes[2] != codes[0]

    def test_rows_sharing_neither_key_are_never_joined(self):
        codes = family_codes(["a", "b", "c"], ["s1", "s2", None])
        assert len(set(codes.tolist())) == 3

    def test_a_missing_seed_never_joins_two_rows(self):
        codes = family_codes(["a", "b"], [None, None])
        assert codes[0] != codes[1]

    def test_the_union_chains_through_a_shared_row(self):
        codes = family_codes(["same text", "same text", "other text"], [None, "s", "s"])
        assert len(set(codes.tolist())) == 1

    def test_whitespace_and_case_variants_are_one_family(self):
        codes = family_codes(["Ask  Me\n", "ask me", "ask you"], [None, None, None])
        assert codes[0] == codes[1]
        assert codes[2] != codes[0]

    def test_family_ids_are_deterministic_across_rebuilds(self):
        prompts = ["a", "b", "a", "c", "d"]
        seeds = [None, "s", None, "s", None]
        np.testing.assert_array_equal(family_codes(prompts, seeds),
                                      family_codes(prompts, seeds))

    def test_an_empty_prompt_refuses_rather_than_joining_unrelated_rows(self):
        with pytest.raises(ValueError, match="empty"):
            family_codes(["", "   "], [None, None])

    def test_the_empty_key_refusal_reads_in_full(self):
        with pytest.raises(ValueError, match=r"^Grouping key is empty\.$"):
            normalised_key("   ")

    def test_the_key_is_the_text_with_its_whitespace_collapsed_and_its_case_folded(self):
        assert normalised_key("Ask  Me\n") == "ask me"

    def test_a_prompt_and_a_seed_carrying_the_same_text_are_not_one_family(self):
        codes = family_codes(["shared", "other"], [None, "shared"])
        assert codes[0] != codes[1]

    def test_a_seed_column_shorter_than_the_prompts_refuses(self):
        with pytest.raises(ValueError, match=r"^Prompts and seeds have different lengths: 2 and 1\.$"):
            family_codes(["a", "b"], [None])

    def test_the_family_codes_are_an_integer_column(self):
        assert family_codes(["a", "b"], [None, None]).dtype == np.int64


class TestRepresentationJoin:
    def test_each_row_carries_its_own_representation(self):
        rows = [a_row("b", "second"), a_row("a", "first")]
        ids, matrix = representations_for(["a", "b"])
        frame = build_frame(rows, ids, matrix)
        np.testing.assert_array_equal(frame["item_id"], ["a", "b"])
        np.testing.assert_array_equal(frame["representation"][0], matrix[0])
        np.testing.assert_array_equal(frame["representation"][1], matrix[1])

    def test_permuting_the_input_row_order_yields_a_byte_identical_frame(self, tmp_path):
        rows = [a_row(f"it{i}", f"prompt {i}", verdict=i % 2, oracle=(i // 2) % 2)
                for i in range(8)]
        ids, matrix = representations_for([r.item_id for r in rows])

        straight = save_frame(build_frame(rows, ids, matrix), tmp_path / "straight.npz")
        order = np.random.default_rng(3).permutation(len(rows))
        permuted = save_frame(build_frame([rows[i] for i in order], ids[order], matrix[order]),
                              tmp_path / "permuted.npz")
        assert straight == permuted

    def test_a_row_without_a_representation_refuses(self):
        ids, matrix = representations_for(["a"])
        with pytest.raises(ValueError, match="missing representations"):
            build_frame([a_row("a", "first"), a_row("b", "second")], ids, matrix)

    def test_a_duplicate_representation_id_refuses(self):
        ids, matrix = representations_for(["a", "a"])
        with pytest.raises(ValueError, match="duplicate"):
            build_frame([a_row("a", "first")], ids, matrix)

    def test_a_duplicate_row_refuses(self):
        ids, matrix = representations_for(["a", "b"])
        with pytest.raises(ValueError, match="duplicate"):
            build_frame([a_row("a", "first"), a_row("a", "again")], ids, matrix)

    def test_a_representation_shorter_than_its_id_column_refuses(self):
        ids, matrix = representations_for(["a", "b"])
        with pytest.raises(ValueError, match="Representation IDs and rows have different lengths"):
            build_frame([a_row("a", "first")], ids, matrix[:1])

    def test_a_non_finite_representation_refuses(self):
        ids, matrix = representations_for(["a"])
        matrix[0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            build_frame([a_row("a", "first")], ids, matrix)

    def test_the_non_finite_refusal_counts_the_offending_values(self):
        ids, matrix = representations_for(["a"])
        matrix[0, 0] = np.inf
        with pytest.raises(ValueError, match=(
                r"^Joined representation contains 1 non-finite values\.$")):
            build_frame([a_row("a", "first")], ids, matrix)

    def test_a_representation_that_is_not_a_matrix_refuses(self):
        with pytest.raises(ValueError, match=(
                r"^Representation must be two-dimensional; got shape \(3,\)\.$")):
            build_frame([a_row("a", "first")], np.array(["a"]), np.arange(3.0))

    def test_the_duplicate_refusal_names_the_column_that_repeated(self):
        ids, matrix = representations_for(["a", "a"])
        with pytest.raises(ValueError, match=r"representation item ids contain 1 duplicate item IDs"):
            build_frame([a_row("a", "first")], ids, matrix)

        ids, matrix = representations_for(["a", "b"])
        with pytest.raises(ValueError, match=r"corpus rows contain 1 duplicate item IDs"):
            build_frame([a_row("a", "first"), a_row("a", "again")], ids, matrix)


class TestWrittenOnceWithDigests:
    def test_rebuilding_from_unchanged_inputs_is_byte_identical(self, tmp_path):
        rows = [a_row(f"it{i}", f"prompt {i}", verdict=i % 2, oracle=(i * 3) % 2)
                for i in range(6)]
        ids, matrix = representations_for([r.item_id for r in rows])
        first = save_frame(build_frame(rows, ids, matrix), tmp_path / "first.npz")
        second = save_frame(build_frame(rows, ids, matrix), tmp_path / "second.npz")
        assert first == second == digest_of(tmp_path / "second.npz")

    def test_saving_twice_to_one_path_refuses(self, tmp_path):
        ids, matrix = representations_for(["a"])
        frame = build_frame([a_row("a", "first")], ids, matrix)
        path = tmp_path / "once.npz"
        save_frame(frame, path)
        with pytest.raises(FileExistsError, match=(
                rf"^Refusing to overwrite existing frame: {re.escape(str(path))}\.$")):
            save_frame(frame, path)

    def test_a_column_that_is_not_a_plain_array_is_refused_rather_than_pickled(self, tmp_path):
        with pytest.raises(ValueError, match="pickle"):
            save_frame({"junk": np.array([{"a": 1}], dtype=object)}, tmp_path / "pickled.npz")

    def test_the_archive_reserves_zip64_headroom_for_members_of_unknown_size(self, tmp_path):
        ids, matrix = representations_for(["a"])
        path = tmp_path / "headroom.npz"
        save_frame(build_frame([a_row("a", "p0")], ids, matrix), path)
        assert ZIP64_EXTRA_FIELD_HEADER in path.read_bytes()

    def test_the_manifest_carries_a_sha256_of_every_input_file(self, tmp_path):
        jsonl = tmp_path / "rows.jsonl"
        jsonl.write_text(json.dumps({"item_id": "a", "prompt": "first"}) + "\n")
        npy = tmp_path / "z.npy"
        np.save(npy, np.zeros((1, 3)))

        manifest = input_manifest([jsonl, npy])
        assert set(manifest) == {str(jsonl), str(npy)}
        for path, entry in manifest.items():
            assert entry["sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest()
            assert entry["bytes"] == len(open(path, "rb").read())


class TestFrameSchema:
    def test_is_stream_is_boolean_and_all_false_for_a_table_one_frame(self):
        ids, matrix = representations_for(["a", "b"])
        frame = build_frame([a_row("a", "first"), a_row("b", "second")], ids, matrix)
        assert frame["is_stream"].dtype == np.bool_
        assert not frame["is_stream"].any()

    def test_every_column_is_aligned_and_the_representation_keeps_its_source_dtype(self):
        ids, matrix = representations_for(["a", "b", "c"])
        frame = build_frame([a_row("a", "p0"), a_row("b", "p1"), a_row("c", "p2")], ids,
                            matrix.astype(np.float32))
        assert frame["representation"].dtype == np.float32
        assert {len(column) for column in frame.values()} == {3}

    def test_a_float32_source_representation_is_stored_float32(self):
        ids, matrix = representations_for(["a", "b"])
        frame = build_frame([a_row("a", "p0"), a_row("b", "p1")], ids,
                            matrix.astype(np.float32))
        assert frame["representation"].dtype == np.float32
        np.testing.assert_array_equal(frame["representation"], matrix.astype(np.float32))

    def test_a_float64_source_representation_is_stored_float64(self):
        ids, matrix = representations_for(["a", "b"])
        frame = build_frame([a_row("a", "p0"), a_row("b", "p1")], ids, matrix)
        assert frame["representation"].dtype == np.float64

    def test_the_saved_frame_round_trips_the_source_dtype(self, tmp_path):
        ids, matrix = representations_for(["a", "b"])
        save_frame(build_frame([a_row("a", "p0"), a_row("b", "p1")], ids,
                               matrix.astype(np.float32)), tmp_path / "narrow.npz")
        assert load_frame(tmp_path / "narrow.npz")["representation"].dtype == np.float32

    def test_a_non_finite_float32_representation_still_refuses(self):
        ids, matrix = representations_for(["a"])
        narrow = matrix.astype(np.float32)
        narrow[0, 0] = np.float32("nan")
        with pytest.raises(ValueError, match="finite"):
            build_frame([a_row("a", "p0")], ids, narrow)

    def test_the_human_column_appears_only_when_every_row_carries_one(self):
        ids, matrix = representations_for(["a", "b"])
        rows = [a_row("a", "p0", human=1), a_row("b", "p1", human=0)]
        assert "human" in build_frame(rows, ids, matrix)
        assert "human" not in build_frame([a_row("a", "p0"), a_row("b", "p1")], ids, matrix)

    def test_a_partly_present_human_column_refuses(self):
        ids, matrix = representations_for(["a", "b"])
        with pytest.raises(ValueError, match="human"):
            build_frame([a_row("a", "p0", human=1), a_row("b", "p1")], ids, matrix)

    def test_a_label_outside_zero_one_refuses(self):
        ids, matrix = representations_for(["a"])
        with pytest.raises(ValueError, match="oracle"):
            build_frame([a_row("a", "p0", oracle=2)], ids, matrix)

    @pytest.mark.parametrize("column", ["verdict", "oracle", "human"])
    def test_the_refusal_names_the_label_column_it_read(self, column):
        ids, matrix = representations_for(["a"])
        with pytest.raises(ValueError, match=(
                rf"^{column} must contain only 0 and 1; found \[2\]\.$")):
            build_frame([a_row("a", "p0", **{column: 2})], ids, matrix)

    def test_the_frame_carries_the_columns_the_package_reads_and_no_others(self):
        ids, matrix = representations_for(["a", "b"])
        frame = build_frame([a_row("a", "p0"), a_row("b", "p1")], ids, matrix)
        assert set(frame) == {"item_id", "representation", "verdict", "oracle", "family",
                              "is_stream"}

    def test_the_human_column_carries_the_corpus_labels(self):
        ids, matrix = representations_for(["a", "b"])
        frame = build_frame([a_row("a", "p0", human=1), a_row("b", "p1", human=0)], ids, matrix)
        np.testing.assert_array_equal(frame["human"], [1, 0])

    def test_the_label_columns_are_integer_whatever_the_corpus_wrote_them_as(self):
        ids, matrix = representations_for(["a", "b"])
        frame = build_frame([a_row("a", "p0", verdict=True, oracle=False),
                             a_row("b", "p1", verdict=False, oracle=True)], ids, matrix)
        assert frame["verdict"].dtype == np.int64
        assert frame["oracle"].dtype == np.int64
        np.testing.assert_array_equal(frame["verdict"], [1, 0])

    def test_the_stream_flag_is_boolean_whatever_the_corpus_wrote_it_as(self):
        ids, matrix = representations_for(["a", "b"])
        rows = [FrameRow(item_id="a", prompt="p0", verdict=0, oracle=0, is_stream=1),
                FrameRow(item_id="b", prompt="p1", verdict=0, oracle=0, is_stream=0)]
        assert build_frame(rows, ids, matrix)["is_stream"].dtype == np.bool_

    def test_the_package_loads_the_saved_frame_and_slices_it(self, tmp_path):
        rows = [a_row("a", "p0", verdict=1, oracle=1), a_row("b", "p1", verdict=0, oracle=1)]
        ids, matrix = representations_for(["a", "b"])
        save_frame(build_frame(rows, ids, matrix), tmp_path / "frame.npz")

        loaded = load_frame(tmp_path / "frame.npz")
        assert loaded["is_stream"].dtype == np.bool_
        np.testing.assert_array_equal(slice_of(loaded, "oracle").agreement, [1, 0])

    def test_a_column_the_package_does_not_read_survives_the_round_trip_untouched(self,
                                                                                  tmp_path):
        ids, matrix = representations_for(["a", "b", "c"])
        frame = build_frame([a_row("a", "p0"), a_row("b", "p1"), a_row("c", "p2")], ids, matrix)
        frame["is_drift_item"] = np.array([False, True, False])
        frame["stream_lambda"] = np.array([0.0, 0.25, 0.5])
        save_frame(frame, tmp_path / "extra.npz")

        loaded = load_frame(tmp_path / "extra.npz")
        assert loaded["is_drift_item"].dtype == np.bool_
        assert loaded["stream_lambda"].dtype == np.float64
        np.testing.assert_array_equal(loaded["is_drift_item"], [False, True, False])
        np.testing.assert_array_equal(loaded["stream_lambda"], [0.0, 0.25, 0.5])

    def test_slicing_a_frame_carries_every_column_including_the_ones_it_does_not_read(self):
        from rcv.frames import concatenated
        from rcv.frames import rows as sliced

        ids, matrix = representations_for(["a", "b", "c"])
        frame = build_frame([a_row("a", "p0"), a_row("b", "p1"), a_row("c", "p2")], ids, matrix)
        frame["stream_lambda"] = np.array([0.0, 0.25, 0.5])

        taken = sliced(frame, np.array([False, True, True]))
        np.testing.assert_array_equal(taken["stream_lambda"], [0.25, 0.5])
        np.testing.assert_array_equal(
            concatenated(taken, taken)["stream_lambda"], [0.25, 0.5, 0.25, 0.5])


class TestTriageRound3Defects:
    def test_a_non_binary_stored_verdict_is_refused_not_truncated(self):
        from rcv.frame_building import build_beaver_verdicts

        costs = np.array([2.0, 4.0])
        truncatable = np.array([0.4, 1.4])
        with pytest.raises(ValueError, match=(
                r"^Stored Beaver verdicts must contain only 0 and 1\.$")):
            build_beaver_verdicts(costs, truncatable)

    def test_a_refused_save_leaves_nothing_at_the_destination(self, tmp_path):
        from rcv.frame_building import save_frame

        destination = tmp_path / "frame.npz"
        poisoned = {"verdict": np.array([0, 1]), "bad": np.array([object(), object()])}
        with pytest.raises(Exception):
            save_frame(poisoned, destination)
        assert not destination.exists(), "a refused save must leave nothing behind"
        save_frame({"verdict": np.array([0, 1])}, destination)
        assert destination.exists()


class TestClassifierScoreColumn:
    def test_the_frame_carries_the_classifier_score_when_given(self):
        from rcv.frame_building import FrameRow, build_frame

        rows = [FrameRow(item_id=f"i{n}", prompt=f"p{n}", verdict=n % 2, oracle=(n + 1) % 2,
                         clf_score=0.1 * n) for n in range(6)]
        ids = np.array([f"i{n}" for n in range(6)])
        matrix = np.arange(12, dtype=np.float32).reshape(6, 2)
        frame = build_frame(rows, ids, matrix)
        assert frame["clf_score"].dtype == np.float64
        assert frame["clf_score"][3] == pytest.approx(0.3)

    def test_a_score_column_of_whole_numbers_is_still_widened(self):
        from rcv.frame_building import FrameRow, build_frame

        rows = [FrameRow(item_id=f"i{n}", prompt=f"p{n}", verdict=n % 2, oracle=(n + 1) % 2,
                         clf_score=n) for n in range(6)]
        ids = np.array([f"i{n}" for n in range(6)])
        matrix = np.arange(12, dtype=np.float32).reshape(6, 2)
        frame = build_frame(rows, ids, matrix)
        assert frame["clf_score"].dtype == np.float64

    def test_a_partial_score_column_is_refused(self):
        from rcv.frame_building import FrameRow, build_frame

        rows = [FrameRow(item_id=f"i{n}", prompt=f"p{n}", verdict=n % 2, oracle=(n + 1) % 2,
                         clf_score=(0.1 * n if n else None)) for n in range(4)]
        ids = np.array([f"i{n}" for n in range(4)])
        matrix = np.arange(8, dtype=np.float32).reshape(4, 2)
        with pytest.raises(ValueError, match="clf_score"):
            build_frame(rows, ids, matrix)
