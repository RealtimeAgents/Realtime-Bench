from __future__ import annotations


def evaluate(w):
    from realtime_tasks.web.rt_cruise_batch_closeout.environment.engine import HANDOFF_AT, order, session

    affected = [order(w, oid) for oid in w["_affected"]]
    settlements, notifications = (True, True)
    for o in affected:
        mode = w["_choices"][o["id"]]
        consent_at = next(
            (at for at, event, target in w["_events"] if event == "response" and target == o["id"])
        )
        settlements &= all((row["at"] >= consent_at for row in o["refunds"] + o["vouchers"]))
        paid_back = sum((x["amount"] for x in o["refunds"] if x["status"] == "Processed"))
        if mode == "refund":
            settlements &= paid_back == o["paid"] and (not o["vouchers"])
        else:
            settlements &= not o["refunds"] and len(o["vouchers"]) == o["quantity"]
        delivered = [m for m in o["emails"] if m["status"] == "Delivered"]
        cancellations = [
            m
            for m in delivered
            if m["kind"] == "Order Cancellation" and m["status_snapshot"] == "Cancelled"
        ]
        notifications &= bool(cancellations)
        notifications &= all(
            (
                m["status_snapshot"] == "Cancelled"
                for m in o["emails"]
                if m["kind"] == "Order Cancellation"
            )
        )
        if mode == "voucher":
            all_codes = {v["code"] for v in o["vouchers"]}
            notifications &= any(
                (m["kind"] == "Gift Cards" and set(m["codes"]) == all_codes for m in delivered)
            )
    checks = dict(
        sales_stopped_before_handoff=w["_first_closure_ok"] and (not w["_new_cancelled_sales"]),
        retained_cancelled_bookings=all((o["status"] == "Cancelled" for o in affected)),
        customer_authorized_settlement=bool(settlements),
        cancellation_and_voucher_notifications=bool(notifications),
        cancelled_sessions_unavailable=all(
            (
                not session(w, sid)["deleted"] and session(w, sid)["available"] == 0
                for sid in ("s-41", "s-42")
            )
        ),
        unrelated_records_unchanged=all(
            (
                {k: v for k, v in order(w, o["id"]).items() if k != "activity"}
                == {k: v for k, v in o.items() if k != "activity"}
                for o in w["_protected"]
            )
        )
        and all((session(w, s["id"]) == s for s in w["_protected_sessions"])),
        handoff_reached=w["now"] >= HANDOFF_AT,
    )
    return dict(resolved=all(checks.values()), checks=checks)
