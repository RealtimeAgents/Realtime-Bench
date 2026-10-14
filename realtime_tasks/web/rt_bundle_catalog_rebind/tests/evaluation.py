from __future__ import annotations


def evaluate(w):
    from realtime_tasks.web.rt_bundle_catalog_rebind.environment.engine import (
        CAMPAIGN_ENDS_AT,
        components_preserved,
        correct_listing,
    )

    purchases = {o["label"]: o for o in w["orders"]}
    first = purchases.get("first")
    last = purchases.get("last")
    checks = {
        "early_order": bool(first and (not first["missed"]) and first["listing_correct"]),
        "current_bundle": correct_listing(w),
        "late_order": bool(last and (not last["missed"]) and last["listing_correct"]),
        "published": w["parent"]["status"] == "active" and w["parent"]["online"],
        "components_preserved": components_preserved(w),
        "other_preserved": w["other"] == w["original_other"],
        "campaign_complete": w["now"] >= CAMPAIGN_ENDS_AT,
    }
    return {"resolved": all(checks.values()), "checks": checks}
