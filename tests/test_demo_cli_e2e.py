from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from importlib import resources
from typing import Any

import pytest

from conftest import REPO_ROOT

GOLDEN = REPO_ROOT / "tests" / "golden" / "demo_transcript.txt"
RECORDS = resources.files("rcv.demo") / "records"
CHAIN_SEEDS = (42, 789, 1024, 456, 123)


def _run(args: list[str], cwd=None, **env_over: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    for key, value in env_over.items():
        env.pop(key, None) if value is None else env.update({key: value})
    return subprocess.run(args, capture_output=True, env=env, cwd=cwd or REPO_ROOT)


def _records() -> Iterator[Any]:
    for seed in CHAIN_SEEDS:
        for path in sorted((RECORDS / f"chain_{seed}").iterdir(), key=lambda p: p.name):
            if path.name.endswith(".json"):
                yield path


def test_module_entry_point_matches_the_golden_bytes() -> None:
    proc = _run([sys.executable, "-m", "rcv", "demo"], RCV_DATA_ROOT=None)
    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout == GOLDEN.read_bytes()


def test_console_script_matches_the_golden_bytes() -> None:
    script = shutil.which("rcv")
    if script is None:
        pytest.skip("the rcv console script is not on PATH")
    proc = _run([script, "demo"], RCV_DATA_ROOT=None)
    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout == GOLDEN.read_bytes()


def _generated_locales() -> set[str]:
    proc = subprocess.run(["locale", "-a"], capture_output=True, text=True)
    return set(proc.stdout.split())


@pytest.mark.parametrize("locale", ["C", "de_DE.UTF-8"])
def test_locale_never_changes_the_bytes(locale: str) -> None:
    if locale != "C" and locale not in _generated_locales():
        pytest.skip(f"{locale} is not generated here, so this run would silently fall back to C")
    proc = _run([sys.executable, "-m", "rcv", "demo"], LC_ALL=locale, LANG=locale)
    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout == GOLDEN.read_bytes()


def test_a_reader_that_closes_the_pipe_early_is_not_a_crash() -> None:
    command = f"set -o pipefail; {sys.executable} -m rcv demo | head -1"
    proc = subprocess.run(["bash", "-c", command], capture_output=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr.decode()
    assert b"BrokenPipeError" not in proc.stderr


def test_the_replay_never_consults_a_data_root() -> None:
    proc = _run([sys.executable, "-m", "rcv", "demo"], RCV_DATA_ROOT="/nonexistent/must-not-matter")
    assert proc.returncode == 0, proc.stderr.decode()
    assert proc.stdout == GOLDEN.read_bytes()


def test_the_replay_writes_nothing(tmp_path) -> None:
    proc = _run([sys.executable, "-m", "rcv", "demo"], cwd=tmp_path)
    assert proc.returncode == 0
    assert list(tmp_path.iterdir()) == []


def test_all_records_are_reachable_as_package_data() -> None:
    assert len(list(_records())) == 22
    assert (RECORDS / "MANIFEST.md").is_file()
