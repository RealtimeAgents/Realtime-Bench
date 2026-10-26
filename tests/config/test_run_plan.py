"""``rtbench run -c PLAN`` (evaluation/harbor_cli.py plan_job): a plan field the Harbor job does
not carry, and a misspelt one, are refused before the run instead of dropped."""

from __future__ import annotations

import pytest

from realtime_framework.evaluation import harbor_cli

PLAN = """\
suite: public
models:
  - my-model
runtime:
  seed: 42
  mode: default
  clock_mode: wall
  fps: 1
  duration_minutes: 5
"""


@pytest.fixture(autouse=True)
def no_dataset(monkeypatch):
    # suite_dataset links task folders under temp/; the job only needs the selected names.
    monkeypatch.setattr(
        harbor_cli, "suite_dataset", lambda suite, selected: {"path": suite, "task_names": selected}
    )


def refused(tmp_path, text, message):
    path = tmp_path / "plan.yaml"
    path.write_text(text)
    with pytest.raises(SystemExit) as caught:
        harbor_cli.plan_job(path)
    assert str(caught.value) == f"{path}: {message}"


@pytest.mark.parametrize(
    "line, field",
    [
        ("  interaction_mod: streaming\n", "interaction_mod"),
        ("  seeed: 7\n", "seeed"),
        ("  graphics_mode: gpu\n", "graphics_mode"),
        ("  recording_encoder: gpu\n", "recording_encoder"),
    ],
)
def test_a_runtime_field_the_job_does_not_carry_is_refused(tmp_path, line, field):
    refused(tmp_path, PLAN + line, f"Unknown runtime fields: [{field!r}]")


@pytest.mark.parametrize(
    "text, field",
    [
        ("execution:\n  lane: 3\n", "execution"),
        ("execution:\n  workers: 8\n", "execution"),
        ("benchmark_id: nightly\n", "benchmark_id"),
    ],
)
def test_a_plan_field_the_job_does_not_carry_is_refused(tmp_path, text, field):
    refused(tmp_path, PLAN + text, f"Unknown benchmark fields: [{field!r}]")


def test_a_protocol_other_than_the_clock_s_is_refused(tmp_path):
    """The trial runs the protocol its clock implies (evaluation/_trial.py)."""
    refused(
        tmp_path,
        PLAN + "  protocol: my-own-protocol\n",
        "runtime.protocol must be 'wall-clock' on the wall clock",
    )
    refused(
        tmp_path,
        PLAN.replace("clock_mode: wall", "clock_mode: token")
        + "  tokens_per_frame: 64\n  protocol: wall-clock\n",
        "runtime.protocol must be 'time-aware-token-v1' on the token clock",
    )


@pytest.mark.parametrize("name", ["default", "paused-clock", "token-clock"])
def test_the_shipped_plans_are_accepted(name):
    (agent,) = harbor_cli.plan_job(harbor_cli.ROOT / f"configs/runs/{name}.yaml")["agents"]
    assert {"suite", "fps", "interaction_mode", "duration_minutes", "seed", "clock"} <= set(
        agent["kwargs"]
    )
