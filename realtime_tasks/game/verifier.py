"""Validate captured game evidence and dispatch task-owned success criteria."""


def state_value(state, path):
    value = state
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def numeric_score(state, field):
    try:
        return float(state_value(state or {}, field))
    except (TypeError, ValueError):
        return None


def score_target(st, *, score_field, target_score, saved_start=None, saved_best=None):
    cur = numeric_score(st, score_field)
    target = float(target_score)
    # progress = (best - start) / (target - start), clamped to [0, 1]; success =
    # target_reached = the peak reached the target (fairer than the live score for games whose
    # score can drop/reset before we read it at the end).
    start_missing = saved_start is None
    best_missing = saved_best is None
    start = saved_start if saved_start is not None else 0.0
    best = saved_best if saved_best is not None else start
    denom = target - start
    progress = 0.0 if denom <= 0 else max(0.0, min(1.0, (best - start) / denom))
    # Safety net against a broken/late baseline: if the episode's FIRST sampled score was
    # already >= target (e.g. an auto-running game that drifted past the goal during warmup, or a
    # reset() that failed to zero the score), the goal was "reached" before the agent acted —
    # that is NOT a real solve. reset_episode() pins the baseline from a fresh episode so this
    # normally holds start≈0 < target; the guard only fires on the pathological inflated baseline.
    baseline_suspect = start >= target
    score_inconsistent = state_value(st, "metrics.score_inconsistent") is True
    auxiliary_violation = state_value(st, "metrics.auxiliary_tool_violation") is True
    invalid_reasons = []
    if cur is None or start_missing or best_missing:
        invalid_reasons.append("score_unavailable")
    if baseline_suspect:
        invalid_reasons.append("baseline_suspect")
    if score_inconsistent:
        invalid_reasons.append("score_inconsistent")
    if auxiliary_violation:
        invalid_reasons.append("auxiliary_tool_violation")
    target_reached = (
        (best is not None) and (best >= target) and (not baseline_suspect) and not invalid_reasons
    )
    return {
        "resolved": bool(target_reached),
        "evaluation_version": 5,
        "progress": round(progress, 4),
        "score": state_value(st, score_field),
        "score_start": start,
        "score_current": cur,
        "score_best": best,
        "target": target,
        "target_reached": bool(target_reached),
        "baseline_suspect": bool(baseline_suspect),
        "score_unavailable": bool(cur is None or start_missing or best_missing),
        "evaluation_valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "score_inconsistent": score_inconsistent,
        "auxiliary_tool_violation": auxiliary_violation,
        "score_field": score_field,
        "terminal": state_value(st, "terminal.isTerminal"),
        "outcome": state_value(st, "terminal.outcome"),
    }


def verify(task_id, state, *, score_field, target_score, saved_start=None, saved_best=None):
    from realtime_tasks.registry import EVALUATION, load_task_module

    evaluator = load_task_module(task_id, EVALUATION, family="game")
    if score_field != evaluator.SCORE_FIELD or target_score != evaluator.TARGET:
        raise ValueError("Game evidence differs from the declared scoring contract")
    return evaluator.evaluate(state, saved_start=saved_start, saved_best=saved_best)
