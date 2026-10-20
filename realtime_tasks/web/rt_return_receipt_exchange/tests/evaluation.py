from __future__ import annotations


def evaluate(w):
    from realtime_tasks.web.rt_return_receipt_exchange.environment.engine import (
        DISPATCH_HANDOVER_AT,
        STOCK_HANDOVER_AT,
        deepcopy,
        first_complete,
    )

    physical = all(
        (
            u["status"] == ("rejected" if u["actual_reason"] else "received")
            and u["reason"] == u["actual_reason"]
            for u in w["units"]
        )
    )
    stock = all(
        (u["stocked"] == ("" if u["actual_reason"] else u["actual_location"]) for u in w["units"])
    )
    release = all((u["released"] == (not bool(u["actual_reason"])) for u in w["units"]))
    early = w["first_met"] if w["first_cutoff"] else first_complete(w)
    first_events = {e["unit"] for e in w["stock_events"] if e["at"] < STOCK_HANDOVER_AT}
    early = early and all((u["id"] in first_events for u in w["units"][: w["first"]]))
    timely = all((e["at"] < DISPATCH_HANDOVER_AT for e in w["stock_events"] + w["release_events"]))
    resolved = bool(
        w["later_posted"]
        and w["paid"]
        and physical
        and stock
        and release
        and early
        and timely
        and (not w["unsafe"])
        and (not w["manual_resolved"])
    )
    return {
        "resolved": resolved,
        "first_handover_met": early,
        "physical_receipt_matches": physical,
        "inventory_matches": stock,
        "replacement_holds_released": release,
        "paid": w["paid"],
        "unsafe_operations": deepcopy(w["unsafe"]),
        "stock_events": deepcopy(w["stock_events"]),
        "release_events": deepcopy(w["release_events"]),
    }
