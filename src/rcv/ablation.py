"""Run routing and feature-source ablations."""

import copy
import json
from pathlib import Path

import yaml

from rcv.runner import run_protocol, write_once

A1_GRID = {
    "neither": (False, False),
    "calibration_only": (False, True),
    "probe_only": (True, False),
    "both": (True, True),
}

ROUTING_FACTOR = "routing: probe and calibration are routed by verdict regime"


def run_routing_factorial(config, output_dir):
    """Run the four probe/calibration routing combinations."""
    if not isinstance(config, dict):
        config = yaml.safe_load(Path(config).read_text())
    output_dir = Path(output_dir)

    cells = {}
    for name, (route_probe, route_calibration) in A1_GRID.items():
        variant = copy.deepcopy(config)
        variant["study"] = f"{config['study']}-a1-{name}"
        variant["estimator"]["route_probe"] = route_probe
        variant["estimator"]["route_calibration"] = route_calibration
        cells[name] = run_protocol(variant, output_dir / name)

    factorial = {
        "cells": cells,
        "routing_contrast": routing_contrast_of(output_dir, cells),
        "provenance": {"config": config},
    }
    write_once(output_dir / "factorial.json", factorial)
    return factorial


def routing_contrast_of(output_dir, cells):
    """Return per-seed deltas between fully routed and pooled cells."""
    both = per_seed_readouts(output_dir / "both", cells["both"]["seeds"])
    neither = per_seed_readouts(output_dir / "neither", cells["neither"]["seeds"])
    contrast = {"factor": ROUTING_FACTOR}
    for metric in ("corrected_adherence", "caught_share"):
        deltas = {str(seed): both[seed][metric] - neither[seed][metric]
                  for seed in cells["both"]["seeds"]}
        values = list(deltas.values())
        contrast[f"per_seed_{metric}_delta"] = deltas
        contrast[f"{metric}_wins_ties_losses"] = [
            sum(delta > 0 for delta in values),
            sum(delta == 0 for delta in values),
            sum(delta < 0 for delta in values),
        ]
    return contrast


def per_seed_readouts(cell_dir, seeds):
    readouts = {}
    for seed in seeds:
        record = json.loads((Path(cell_dir) / f"seed_{seed}.json").read_text())
        readouts[seed] = record["readout"]
    return readouts


def run_probe_family_pairs(config, output_dir, family_specs):
    """Run routed-versus-pooled comparisons for each probe family."""
    if not isinstance(config, dict):
        config = yaml.safe_load(Path(config).read_text())
    output_dir = Path(output_dir)

    cells = {}
    for spec in family_specs:
        for routed in (False, True):
            name = f"{spec['family']}_{'routed' if routed else 'pooled'}"
            variant = copy.deepcopy(config)
            variant["study"] = f"{config['study']}-a2-{name}"
            variant["estimator"]["probe"] = dict(spec)
            variant["estimator"]["route_probe"] = routed
            variant["estimator"]["route_calibration"] = routed
            cells[name] = run_protocol(variant, output_dir / name)

    contrasts = {}
    for spec in family_specs:
        family = spec["family"]
        routed = per_seed_readouts(output_dir / f"{family}_routed",
                                   cells[f"{family}_routed"]["seeds"])
        pooled = per_seed_readouts(output_dir / f"{family}_pooled",
                                   cells[f"{family}_pooled"]["seeds"])
        contrast = {"factor": ROUTING_FACTOR}
        for metric in ("corrected_adherence", "caught_share"):
            deltas = {str(seed): routed[seed][metric] - pooled[seed][metric]
                      for seed in cells[f"{family}_routed"]["seeds"]}
            values = list(deltas.values())
            contrast[f"per_seed_{metric}_delta"] = deltas
            contrast[f"{metric}_wins_ties_losses"] = [
                sum(delta > 0 for delta in values),
                sum(delta == 0 for delta in values),
                sum(delta < 0 for delta in values),
            ]
        contrasts[family] = contrast

    pairs = {"cells": cells, "family_contrasts": contrasts,
             "provenance": {"config": config, "family_specs": family_specs}}
    write_once(output_dir / "family_pairs.json", pairs)
    return pairs


def run_feature_source_pairs(config, output_dir, sources=("representation", "clf_score")):
    """Run feature-source cells and contrast the first two sources."""
    if not isinstance(config, dict):
        config = yaml.safe_load(Path(config).read_text())
    output_dir = Path(output_dir)

    cells = {}
    for source in sources:
        variant = copy.deepcopy(config)
        variant["study"] = f"{config['study']}-a4-{source}"
        variant["estimator"]["feature_source"] = source
        cells[source] = run_protocol(variant, output_dir / source)

    internal = per_seed_readouts(output_dir / sources[0], cells[sources[0]]["seeds"])
    score = per_seed_readouts(output_dir / sources[1], cells[sources[1]]["seeds"])
    contrast = {"factor": f"probe feature source: {sources[0]} versus {sources[1]}"}
    for metric in ("corrected_adherence", "caught_share"):
        deltas = {str(seed): internal[seed][metric] - score[seed][metric]
                  for seed in cells[sources[0]]["seeds"]}
        values = list(deltas.values())
        contrast[f"per_seed_{metric}_delta"] = deltas
        contrast[f"{metric}_wins_ties_losses"] = [
            sum(delta > 0 for delta in values),
            sum(delta == 0 for delta in values),
            sum(delta < 0 for delta in values),
        ]
    pairs = {"cells": cells, "source_contrast": contrast,
             "provenance": {"config": config, "sources": list(sources)}}
    write_once(output_dir / "feature_source_pairs.json", pairs)
    return pairs
