"""Shared evidence checks and task-owned Web scoring dispatch."""

from realtime_tasks.registry import EVALUATION, load_task_module


def evaluate_engine(engine, episode):
    task_id = getattr(getattr(engine, "spec", None), "name", None) or engine.name
    result = load_task_module(task_id, EVALUATION).evaluate(engine, episode)
    if not isinstance(result, dict) or type(result.get("resolved")) is not bool:
        raise ValueError(f"{task_id}: scoring requires a boolean resolved field")
    return result
