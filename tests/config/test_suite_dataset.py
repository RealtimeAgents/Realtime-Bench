"""``rtbench run`` (evaluation/harbor_cli.py suite_dataset): concurrent jobs on other lanes that
select as many tasks share one folder of task links, so a job finds a link another job has just
made."""

from __future__ import annotations

from pathlib import Path

from realtime_framework.evaluation import harbor_cli


def test_a_link_another_job_made_meanwhile_is_kept(monkeypatch, tmp_path):
    tasks = tmp_path / "realtime_tasks"
    tasks.mkdir()
    (tasks / "sets").symlink_to(harbor_cli.ROOT / "realtime_tasks/sets")
    for task, family in harbor_cli.suite_tasks("public").items():
        (tasks / family / task).mkdir(parents=True)
        (tasks / family / task / "task.toml").touch()
    monkeypatch.setattr(harbor_cli, "ROOT", tmp_path)
    folder = tmp_path / "temp/harbor-suites/public/partial (1)"
    source = tasks / "game/gomoku"
    unlink = Path.unlink

    def unlink_then_other_job_links(self, missing_ok=False):
        # Between this job's unlink and its symlink, the other job links the same task.
        unlink(self, missing_ok=missing_ok)
        if self == folder / "gomoku":
            self.symlink_to(source, target_is_directory=True)

    monkeypatch.setattr(Path, "unlink", unlink_then_other_job_links)

    dataset = harbor_cli.suite_dataset("public", ["gomoku"])

    assert dataset == {"path": str(folder), "task_names": ["gomoku"]}
    assert (folder / "gomoku").resolve() == source
