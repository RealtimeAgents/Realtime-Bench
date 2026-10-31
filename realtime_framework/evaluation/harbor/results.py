"""Read-only presentation of Harbor jobs and benchmark trial outcomes."""

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from realtime_framework.evaluation.records.reporting import active_interval
from realtime_framework.evaluation.records.reporting import episode_outcome

OUTCOME_STATUS = {"pass": "Pass", "fail": "Fail", "timeout": "Timeout", "error": "Error"}


def _read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def job_label(job_dir):
    collection = _read(job_dir / "collection.json")
    config = _read(job_dir / "config.json")
    agents = config.get("agents") or []
    pairs = []
    if collection.get("model"):
        pairs.append((collection["model"], collection.get("reasoning_effort")))
    else:
        for agent in agents:
            kwargs = agent.get("kwargs") or {}
            model = kwargs.get("catalog_model") or agent.get("model_name")
            if model:
                pairs.append((model.split("/", 1)[-1], kwargs.get("reasoning_effort")))
    labels = list(dict.fromkeys(
        f"{model} · reasoning effort {effort if effort is not None else 'default'}"
        for model, effort in pairs
    ))
    return " / ".join(labels) or job_dir.name


def trial_result(job_dir, trial_dir):
    result = _read(trial_dir / "result.json")
    config = result.get("config") or _read(trial_dir / "config.json")
    metadata = (result.get("agent_result") or {}).get("metadata") or {}
    evidence = trial_dir / "agent/benchmark"
    summary = _read(evidence / "run_summary.json")
    active = _read(evidence / "active-loop.json") or metadata.get("active_loop") or {}
    duration = None
    start, end = active.get("start_unix_s"), active.get("end_unix_s")
    if _number(start) and _number(end) and end >= start:
        duration = end - start
    elif _number(start) and end is None and not result.get("finished_at"):
        duration = max(0, datetime.now(UTC).timestamp() - start)
    if duration is None:
        for value in (active.get("active_wall_time_s"), summary.get("loop_time_s"),
                      metadata.get("loop_time_s")):
            if _number(value) and value >= 0:
                duration = value
                break
    # Without recorded loop timing, the trace can recover the scored loop. Harbor's agent_execution
    # also includes sandbox setup, so leave duration unknown when no benchmark timing survives.
    if duration is None:
        try:
            start, end, _ = active_interval(evidence)
            duration = end - start
        except (OSError, ValueError, KeyError):
            pass

    error = (result.get("exception_info") or {}).get("exception_type")
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    reward = reward if _number(reward) else None
    # The runner decides the outcome; the verifier copies it and rewards only a verified pass.
    # Timeout means the model used the whole horizon unsolved, never an infrastructure timeout.
    error_reason = None
    if not result.get("finished_at") and not error:
        status = "Running" if result.get("started_at") else "Pending"
    else:
        verdict = _read(trial_dir / "verifier/evaluation.json")
        if verdict.get("outcome") in OUTCOME_STATUS:
            outcome, error_reason = verdict["outcome"], verdict.get("error_reason")
        elif summary:
            outcome_record = episode_outcome(summary)
            outcome, error_reason = outcome_record["outcome"], outcome_record["error_reason"]
        else:
            outcome, error_reason = "error", None
        if outcome != "error" and (reward is None or error):
            # A runner verdict without a verified reward is never presented as scored.
            outcome, error_reason = "error", None
        if outcome == "error" and not error_reason:
            error_reason = f"harbor_{error}" if error else "unverified_trial"
        status = OUTCOME_STATUS[outcome]
    # Errors are scored as failures; only running and pending trials are unscored.
    passed = status == "Pass" if status in {"Pass", "Fail", "Timeout", "Error"} else None
    agent = result.get("agent_info") or {}
    model = agent.get("model_info") or {}
    agent_config = config.get("agent") or {}
    provider, _, model_name = (agent_config.get("model_name") or "").partition("/")
    if not model_name:
        model_name, provider = provider, ""
    task = result.get("task_name") or (config.get("task") or {}).get("name")
    task = task or Path((config.get("task") or {}).get("path") or trial_dir.name).name
    source = result.get("source") or config.get("source")
    segments = ["jobs", job_dir.name, "tasks", source or "_",
                agent.get("name") or agent_config.get("name") or "_",
                model.get("provider") or provider or "_",
                model.get("name") or model_name or "_", task, "trials", trial_dir.name]
    return {
        "name": trial_dir.name, "task": task,
        "model": model.get("name") or model_name or "Unknown model",
        "effort": metadata.get("reasoning_effort", (agent_config.get("kwargs") or {}).get(
            "reasoning_effort"
        )),
        "run": (metadata.get("collection_member") or {}).get("run"),
        "status": status, "error_reason": error_reason if status == "Error" else None,
        "passed": passed, "reward": reward, "duration_s": duration,
        "href": "/" + "/".join(quote(str(segment), safe="") for segment in segments),
    }


def job_results(job_dir):
    trials = [trial_result(job_dir, path) for path in sorted(job_dir.iterdir())
              if path.is_dir() and ((path / "config.json").is_file()
                                    or (path / "result.json").is_file())]
    groups = defaultdict(list)
    for trial in trials:
        groups[trial["task"]].append(trial)
    tasks = []
    for task, attempts in sorted(groups.items()):
        scored = [trial for trial in attempts if trial["passed"] is not None]
        passes = sum(trial["passed"] for trial in scored)
        tasks.append({"name": task, "trials": attempts, "passes": passes,
                      "scored": len(scored),
                      "pass_rate": passes / len(scored) if scored else None})
    scored = [trial for trial in trials if trial["passed"] is not None]
    return {
        "name": job_dir.name, "display_name": job_label(job_dir),
        "average_pass_rate": sum(trial["passed"] for trial in scored) / len(scored)
        if scored else None,
        "scored": len(scored), "passes": sum(trial["passed"] for trial in scored),
        "trials": trials, "tasks": tasks,
    }
