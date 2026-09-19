import importlib.util
import json
import shutil

import pytest

from conftest import REPO_ROOT


@pytest.fixture(scope="module")
def table1_module():
    path = REPO_ROOT / "smoke/print_table1.py"
    spec = importlib.util.spec_from_file_location("print_table1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copied_bundle(tmp_path):
    shutil.copytree(REPO_ROOT / "results", tmp_path / "results")
    return tmp_path


def test_shipped_records_reproduce_arxiv_table1(table1_module):
    summaries = table1_module.verify_table1(REPO_ROOT)
    assert set(summaries) == set(table1_module.EXPECTED_CELLS)
    assert all(summary["n"] == 10 for summary in summaries.values())


def test_a_missing_cell_is_refused(table1_module, tmp_path):
    bundle = copied_bundle(tmp_path)
    shutil.rmtree(bundle / table1_module.STEERING_RELATIVE / "pku_beaver")

    with pytest.raises(table1_module.TableVerificationError, match="cell directories"):
        table1_module.verify_table1(bundle)


def test_a_missing_seed_is_refused(table1_module, tmp_path):
    bundle = copied_bundle(tmp_path)
    (bundle / table1_module.STEERING_RELATIVE / "pku_beaver/seed_42.json").unlink()

    with pytest.raises(table1_module.TableVerificationError, match="seed files"):
        table1_module.verify_table1(bundle)


def test_a_seed_identity_must_match_its_filename(table1_module, tmp_path):
    bundle = copied_bundle(tmp_path)
    path = bundle / table1_module.STEERING_RELATIVE / "pku_beaver/seed_42.json"
    record = json.loads(path.read_text())
    record["seed"] = 123
    path.write_text(json.dumps(record))

    with pytest.raises(table1_module.TableVerificationError, match="records seed 123"):
        table1_module.verify_table1(bundle)


def test_a_changed_result_is_refused(table1_module, tmp_path):
    bundle = copied_bundle(tmp_path)
    path = bundle / table1_module.STEERING_RELATIVE / "pku_beaver/seed_42.json"
    record = json.loads(path.read_text())
    record["readouts"]["oracle"]["caught_share"] += 0.1
    path.write_text(json.dumps(record))

    with pytest.raises(table1_module.TableVerificationError, match="aggregate caught_share"):
        table1_module.verify_table1(bundle)
