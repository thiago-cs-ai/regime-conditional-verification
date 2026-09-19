from __future__ import annotations

import importlib

import pytest

MODULES = [
    "rcv.extraction",
    "rcv.extraction.templates",
    "rcv.extraction._lg3_context",
    "rcv.extraction._io",
    "rcv.extraction.lg3_extract",
    "rcv.extraction.wg_extract",
    "rcv.extraction.beaver_stub",
    "rcv.extraction.beaver_extract",
    "rcv.extraction.finetune",
    "rcv.extraction.build_joined",
    "rcv.extraction.diff_profile",
]


@pytest.mark.parametrize("mod", MODULES)
def test_every_extraction_module_imports(mod):
    importlib.import_module(mod)


def test_lg3_extract_dry_run(tmp_path, capsys):
    from rcv.extraction import lg3_extract

    rc = lg3_extract.main(
        ["--input", str(tmp_path / "in.jsonl"), "--output", str(tmp_path / "o.jsonl"),
         "--embeddings", str(tmp_path / "z.npy"), "--dry-run"]
    )
    assert rc == 0
    assert "decision_L-1" in capsys.readouterr().out


def test_wg_extract_dry_run(tmp_path, capsys):
    from rcv.extraction import wg_extract

    rc = wg_extract.main(
        ["--input", str(tmp_path / "in.jsonl"), "--output", str(tmp_path / "o.jsonl"),
         "--embeddings", str(tmp_path / "z.npy"), "--dry-run"]
    )
    assert rc == 0
    assert "two-pass" in capsys.readouterr().out


def test_beaver_extract_dry_run(tmp_path, capsys):
    from rcv.extraction import beaver_extract

    rc = beaver_extract.main(
        ["--input", str(tmp_path / "in.jsonl"), "--output", str(tmp_path / "o.jsonl"),
         "--embeddings", str(tmp_path / "z.npy"), "--dry-run"]
    )
    assert rc == 0
    assert "T_xml" in capsys.readouterr().out


def test_finetune_dry_run(tmp_path, capsys):
    from rcv.extraction import finetune

    rc = finetune.main(
        ["--seed", "42", "--corpus-dir", str(tmp_path / "corpus"), "--out", str(tmp_path / "out"),
         "--dry-run"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "seed                       = 42" in out
    assert "decision-token-only CE" in out


def test_build_joined_dry_run(tmp_path, capsys):
    from rcv.extraction import build_joined

    rc = build_joined.main(
        ["--pulled", str(tmp_path), "--out-dir", str(tmp_path / "joined"), "--dry-run"]
    )
    assert rc == 0
    assert "zero z_error flags" in capsys.readouterr().out


def test_diff_profile_dry_run(tmp_path, capsys):
    from rcv.extraction import diff_profile

    rc = diff_profile.main(["--ft-arrays", str(tmp_path / "ft.npz"), "--out", str(tmp_path), "--dry-run"])
    assert rc == 0
    assert "UNSAFE" in capsys.readouterr().out
