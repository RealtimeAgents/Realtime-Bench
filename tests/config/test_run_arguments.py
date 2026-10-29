"""``rtbench run -c JOB -m MODEL`` (evaluation/harbor_cli.py run_arguments): ``-m`` names the
model the job runs, also over a job configuration a run saved, whose agent names the preset it
resolved to (describe_agent's ``kwargs.catalog_model``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from realtime_framework.evaluation import harbor_cli


@pytest.fixture(autouse=True)
def configs_under_tmp(tmp_path, monkeypatch):
    # run_arguments writes the job it builds under ROOT/temp/harbor-configs.
    monkeypatch.setattr(harbor_cli, "ROOT", tmp_path)


def job(arguments: list[str]) -> dict:
    flags = harbor_cli.run_arguments(arguments)
    assert flags[0] == "-c"
    return yaml.safe_load(Path(flags[1]).read_text())


def saved_job(tmp_path, model: str) -> Path:
    """The job configuration Harbor saves (config.json) of a run of ``model``."""
    first = tmp_path / "first.yaml"
    first.write_text(yaml.safe_dump({"tasks": [{"path": "synthetic-task"}]}))
    saved = tmp_path / "config.json"
    saved.write_text(json.dumps(job(["-c", str(first), "-m", model])))
    return saved


@pytest.mark.parametrize(
    "model, model_name, preset",
    [
        ("gpt-6-astra", "openai/gpt-6-astra", "gpt-6-astra"),
        ("openrouter/vendor/some-model", "openrouter/vendor/some-model", None),
    ],
)
def test_m_runs_its_model_over_a_saved_job_of_another(tmp_path, model, model_name, preset):
    saved = saved_job(tmp_path, "gpt-5.6-sol")
    assert json.loads(saved.read_text())["agents"][0]["kwargs"]["catalog_model"] == "gpt-5.6-sol"

    (agent,) = job(["-c", str(saved), "-m", model])["agents"]
    assert agent["model_name"] == model_name
    assert agent["kwargs"].get("catalog_model") == preset
