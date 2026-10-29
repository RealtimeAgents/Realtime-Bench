"""``rtbench upload`` (evaluation/submission.py main): a rerun resumes the archive of the results
it just checked, and packs again the results of jobs replaced at the same paths."""

from __future__ import annotations

import json
import tarfile

import pytest

from realtime_framework.evaluation import submission
from realtime_framework.providers import registry


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """Upload attempts, each the model and evidence of the archive it sent; the first fails."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry, "load_project_secrets", lambda: None)
    sent = []

    def upload(archive, manifest, client, log=print):
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            sent.append({
                "model": json.load(tar.extractfile("submission.json"))["runs"][0]["report"]["model"],
                "evidence": tar.extractfile(next(n for n in names if n.endswith("result.txt")))
                .read().decode(),
                "archive": archive.stat().st_mtime_ns,
            })
        if len(sent) == 1:
            raise submission.SubmissionError("PUT /s?offset=0: connection reset")
        return {}

    monkeypatch.setattr(submission, "upload", upload)
    return sent


def finished(tmp_path, monkeypatch, model, evidence):
    """Three job directories holding ``evidence``, whose reports name ``model``."""
    jobs = [tmp_path / f"my-model-run{index}" for index in (1, 2, 3)]
    for job in jobs:
        job.mkdir(exist_ok=True)
        (job / "result.txt").write_text(evidence)
    report = {"model": model, "outcomes": {"pass": 1}, "expected": 20}
    monkeypatch.setattr(submission, "check_runs", lambda checked: [dict(report) for _ in checked])
    return jobs


def test_a_rerun_resumes_the_archive_of_the_same_results(tmp_path, monkeypatch, uploads):
    jobs = finished(tmp_path, monkeypatch, "my-model", "result")
    with pytest.raises(submission.SubmissionError):
        submission.main(jobs, token="token")
    submission.main(jobs, token="token")
    assert uploads[0] == uploads[1]


def test_jobs_replaced_at_their_paths_are_packed_again(tmp_path, monkeypatch, uploads):
    with pytest.raises(submission.SubmissionError):
        submission.main(finished(tmp_path, monkeypatch, "old-model", "old result"), token="token")
    submission.main(finished(tmp_path, monkeypatch, "new-model", "new result"), token="token")
    assert [(sent["model"], sent["evidence"]) for sent in uploads] == [
        ("old-model", "old result"),
        ("new-model", "new result"),
    ]
