"""The maintainers' private report (evaluation/harbor.py aggregate with official=True) scores only
a released private suite run on a released harness (phase2.md: 私有套件的正式报告); a harness
whose hash differs from the one pyproject.toml locks is ``+dev`` (evaluation/config.py
harness_version). A self-reported report takes any harness, as before."""

from __future__ import annotations

import json

import pytest
import yaml

from realtime_framework.evaluation import harbor

WEB = "rt_accessible_trip_dispatch"
SUITE_HASH = "0" * 64


def _job(tmp_path, monkeypatch, harness_version: str):
    """A one-task job of a released private suite, whose manifest the report finds in a checkout
    of its own, and whose trial ran on ``harness_version``."""
    sets = tmp_path / "checkout" / "realtime_tasks" / "sets"
    sets.mkdir(parents=True)
    manifest = {
        "suite_id": "private",
        "version": "1.0",
        "suite_hash": SUITE_HASH,
        "categories": [{"public_category": "web", "task_ids": [WEB]}],
    }
    (sets / "private.yaml").write_text(yaml.safe_dump(manifest))
    monkeypatch.setattr(harbor, "ROOT", tmp_path / "checkout")
    task = {
        "suite_id": "private", "suite_version": "1.0", "suite_hash": SUITE_HASH,
        "harness_version": harness_version, "harness_hash": "1" * 64, "task_id": WEB,
        "category": "web", "fps": 1.0, "model_input_fps": 1.0,
        "interaction_mode": "turn_based", "duration_minutes": 5.0, "seed": 42,
    }  # fmt: skip
    request = {
        "model": "gemini-3.8-flash", "mode": "default", "model_revision": None, "clock": "wall",
        "task": task, "suite_tasks": {WEB: "web"},
    }  # fmt: skip
    trial = tmp_path / "job" / f"{WEB}__trial" / "agent"
    trial.mkdir(parents=True)
    (trial / "request.json").write_text(json.dumps(request))
    return tmp_path / "job"


def test_an_official_report_refuses_an_unreleased_harness(tmp_path, monkeypatch):
    job = _job(tmp_path, monkeypatch, "1.0+dev")
    with pytest.raises(ValueError) as refused:
        harbor.aggregate(job, official=True)
    assert str(refused.value) == (
        "Official evaluation requires a released harness, not harness@1.0+dev, whose sources "
        "differ from the hash pyproject.toml locks"
    )


def test_an_official_report_takes_a_released_harness(tmp_path, monkeypatch):
    report = harbor.aggregate(_job(tmp_path, monkeypatch, "1.0"), official=True)
    assert (report["evaluation"], report["harness_version"]) == ("maintainer_private", "1.0")


def test_a_self_reported_report_takes_an_unreleased_harness(tmp_path, monkeypatch):
    report = harbor.aggregate(_job(tmp_path, monkeypatch, "1.0+dev"))
    assert (report["evaluation"], report["harness_version"]) == ("self_reported", "1.0+dev")
