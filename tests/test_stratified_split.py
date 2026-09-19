import numpy as np
import pytest

from rcv.splitting import draw_group_blocked_split


def wgmix_like(rare_rows=6, total=200, seed=0):
    family = np.arange(total)
    strata = np.zeros(total, dtype=int)
    strata[np.random.default_rng(seed).choice(total, rare_rows, replace=False)] = 3
    return family, strata


class TestStratifiedSingletonFamilies:
    def test_every_split_carries_the_rare_stratum_on_every_seed(self):
        family, strata = wgmix_like()
        for seed in range(10):
            masks = draw_group_blocked_split(family, 0.2, 0.25, seed, strata=strata)
            for name, mask in masks.items():
                assert (strata[mask] == 3).sum() >= 1, (
                    f"seed {seed}: the {name} split lost the rare stratum")

    def test_the_three_masks_still_partition_every_row(self):
        family, strata = wgmix_like()
        masks = draw_group_blocked_split(family, 0.2, 0.25, seed=1, strata=strata)
        one_each = sum(mask.astype(int) for mask in masks.values())
        assert (one_each == 1).all()

    def test_the_seed_still_governs(self):
        family, strata = wgmix_like()
        first = draw_group_blocked_split(family, 0.2, 0.25, seed=1, strata=strata)
        again = draw_group_blocked_split(family, 0.2, 0.25, seed=1, strata=strata)
        moved = draw_group_blocked_split(family, 0.2, 0.25, seed=2, strata=strata)
        np.testing.assert_array_equal(first["evaluation"], again["evaluation"])
        assert not np.array_equal(first["evaluation"], moved["evaluation"])

    def test_overall_fractions_are_still_honoured(self):
        family, strata = wgmix_like()
        masks = draw_group_blocked_split(family, 0.2, 0.25, seed=1, strata=strata)
        assert masks["evaluation"].sum() == pytest.approx(40, abs=3)
        assert masks["calibration"].sum() == pytest.approx(40, abs=3)


class TestMixedFamiliesStayWhole:
    def test_a_family_with_mixed_strata_moves_whole(self):
        family = np.repeat(np.arange(50), 2)
        strata = np.tile([0, 3], 50)
        masks = draw_group_blocked_split(family, 0.2, 0.25, seed=1, strata=strata)
        for mask in masks.values():
            for one_family in np.unique(family):
                rows_of_family = mask[family == one_family]
                assert rows_of_family.all() or not rows_of_family.any()

    def test_without_strata_the_behaviour_is_the_existing_one(self):
        family = np.repeat(np.arange(40), 3)
        with_none = draw_group_blocked_split(family, 0.3, 0.25, seed=1, strata=None)
        plain = draw_group_blocked_split(family, 0.3, 0.25, seed=1)
        for name in plain:
            np.testing.assert_array_equal(with_none[name], plain[name])


class TestStrataValidation:
    def test_strata_length_must_match(self):
        family = np.arange(10)
        with pytest.raises(ValueError, match="strata"):
            draw_group_blocked_split(family, 0.2, 0.25, seed=1, strata=np.zeros(9, dtype=int))
