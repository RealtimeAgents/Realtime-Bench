from __future__ import annotations


def evaluate(w):
    from realtime_tasks.web.rt_document_revision_routing.environment.engine import (
        CLIENT_SIGNING_AT,
        EPISODE_END,
        SEND_DEADLINE,
        _revision,
        _safe_routing,
        _same_values,
    )

    revised = [r for r in w["_revisions"] if r["number"] > 1]
    r = _revision(w)
    checks = dict(
        withdrawn_before_old_completion=w["_withdrawn_at"] is not None
        and w["_withdrawn_at"] < CLIENT_SIGNING_AT
        and (not w["_old_completed"]),
        only_confirmed_suitable_terms=bool(revised)
        and all((r["sent"] >= 130 and _same_values(r["variables"], w["_target"]) for r in revised)),
        revision_sent_in_time=bool(r and r["number"] > 1 and (r["sent"] <= SEND_DEADLINE)),
        actual_notification_order=bool(revised) and all((_safe_routing(r) for r in revised)),
        fresh_signatures_complete=bool(
            r
            and r["number"] > 1
            and (w["status"] == "Completed")
            and all(
                (
                    r["recipients"][i]["signed"] is not None
                    and r["recipients"][i]["signed"] <= EPISODE_END
                    for i in ("lead", "client")
                )
            )
        ),
        accounts_copy_delivered=bool(
            r
            and r["number"] > 1
            and (r["recipients"]["accounts"]["notified"] is not None)
            and (r["recipients"]["accounts"]["notified"] <= EPISODE_END)
        ),
    )
    return {"resolved": all(checks.values()), "checks": checks, "revision": w["revision"]}
