"""Guard that the test suite does not write Lab artifacts into the repository."""

from pathlib import Path

from tests.test_lab_multi_metric import _make_lab

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tests_run_outside_repo_root(tmp_path):
    assert Path.cwd() == tmp_path
    assert Path.cwd() != REPO_ROOT


def test_lab_artifacts_are_created_in_temp_dir(tmp_path):
    lab = _make_lab(multi_metric=False)

    assert (tmp_path / lab.name / "pipelines").is_dir()
    assert not (REPO_ROOT / lab.name).exists()
