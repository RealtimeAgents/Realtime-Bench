"""ParcelTurn: unit-level receipt, physical stock and paid replacement release."""

from copy import deepcopy

import hashlib

import itertools

import random

TASK_ID = "rt_return_receipt_exchange"

STALE_MESSAGE = "This return changed. Review the current record and try again."

CLOCK_FIELDS = ("clock_seconds", "stock_handover_seconds", "dispatch_handover_seconds")

STOCK_HANDOVER_AT = 120

DISPATCH_HANDOVER_AT = 300

REASONS = [
    "Not in original condition",
    "Item damaged or worn",
    "Wrong item returned",
    "Missing components or accessories",
    "Signs of use or wear",
    "Return window expired",
    "Warehouse rejected",
    "Other",
]

WAREHOUSES = ["East warehouse", "West warehouse"]


def stamp(t):
    seconds = int(16 * 3600 + 55 * 60 + t)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def log(w, text, t, author="ParcelTurn"):
    w["timeline"].append({"time": stamp(t), "author": author, "text": text})


def create_world(seed):
    rng = random.Random(int(hashlib.sha256(str(seed).encode()).hexdigest(), 16))
    first = rng.choice([1, 2])
    first_location = rng.choice(WAREHOUSES)
    later_location = rng.choice(WAREHOUSES)
    bad = rng.choice([0, 1])
    reason = rng.choice(REASONS[1:3]) if bad else ""
    w = {
        "now": 0.0,
        "stage": "Waiting for Return",
        "paid": False,
        "first": first,
        "later_at": rng.randint(130, 160),
        "paid_at": rng.randint(110, 185),
        "later_posted": False,
        "first_cutoff": False,
        "first_met": False,
        "archived": False,
        "manual_resolved": False,
        "unsafe": [],
        "timeline": [],
        "notes": "Size exchange · customer selected Large",
        "units": [],
        "stock_events": [],
        "release_events": [],
        "receipt_events": [],
    }
    for i in range(4):
        w["units"].append(
            {
                "id": f"u{i + 1}",
                "status": "waiting",
                "reason": "",
                "stocked": "",
                "released": False,
                "arrived_at": 0 if i < first else w["later_at"],
                "actual_location": first_location if i < first else later_location,
                "actual_reason": reason if bad and i == 3 else "",
            }
        )
    log(
        w,
        "Return approved. Four Field shirts, Olive / M, exchanged for Olive / L. Return-shipping fee: $9.50.",
        -450,
    )
    log(w, "Payment invoice for $9.50 sent to Jamie Bell.", -420)
    log(
        w,
        f"Parcel 1 inspected at {first_location}: {first} Field shirt{'s' if first != 1 else ''}, Olive / M. All in original condition, ready for shelf stock. The remaining {4 - first} units are in a separate parcel that has not arrived.",
        -20,
        "Nora · Warehouse",
    )
    return w


def first_complete(w):
    return all(u["stocked"] == u["actual_location"] for u in w["units"][: w["first"]])


def advance(w, elapsed):
    t = max(w["now"], float(elapsed))
    events = []
    if not w["first_cutoff"] and t >= STOCK_HANDOVER_AT:
        events.append((STOCK_HANDOVER_AT, "cutoff"))
    if not w["later_posted"] and t >= w["later_at"]:
        events.append((w["later_at"], "parcel"))
    if not w["paid"] and t >= w["paid_at"]:
        events.append((w["paid_at"], "payment"))
    for at, event in sorted(events):
        if event == "cutoff":
            w["first_cutoff"], w["first_met"] = True, first_complete(w)
        elif event == "payment":
            w["paid"] = True
            log(w, "Customer paid the exchange balance of $9.50.", at)
        else:
            w["later_posted"] = True
            later = w["units"][w["first"] :]
            bad = [u for u in later if u["actual_reason"]]
            detail = (
                f"{len(later) - len(bad)} in original condition, ready for shelf stock; 1 rejected — {bad[0]['actual_reason'].lower()}."
                if bad
                else f"All {len(later)} in original condition, ready for shelf stock."
            )
            log(
                w,
                f"Parcel 2 inspected at {later[0]['actual_location']}: {len(later)} units physically received for the Field shirt return. {detail} No further units are outstanding.",
                at,
                "Nora · Warehouse",
            )
    w["now"] = t


def group_key(u):
    return "|".join([u["status"], u["reason"], u["stocked"]])


def groups(w):
    grouped = {}
    for u in w["units"]:
        key = group_key(u)
        g = grouped.setdefault(
            key,
            {
                "id": key,
                "status": u["status"],
                "reason": u["reason"],
                "stocked": u["stocked"],
                "quantity": 0,
            },
        )
        g["quantity"] += 1
    return list(grouped.values())


def releasable(w):
    return [u for u in w["units"] if u["status"] == "received" and not u["released"]]


def public_world(w):
    now = int(w["now"])
    return {
        "rma": "R-2841",
        "order": "#10482",
        "stage": w["stage"],
        "paid": w["paid"],
        "clock_seconds": 16 * 3600 + 55 * 60 + now,
        "stock_handover_seconds": max(0, STOCK_HANDOVER_AT - now),
        "dispatch_handover_seconds": max(0, DISPATCH_HANDOVER_AT - now),
        "fee": 9.5,
        "groups": groups(w),
        "released": sum(u["released"] for u in w["units"]),
        "pending": sum(not u["released"] and u["status"] != "rejected" for u in w["units"]),
        "release_quantity": len(releasable(w)),
        "warehouses": list(WAREHOUSES),
        "reasons": list(REASONS),
        "timeline": deepcopy(w["timeline"]),
        "notes": w["notes"],
        "archived": w["archived"],
        "manual_resolved": w["manual_resolved"],
    }


def unacceptable(w, t):
    return sum(
        u["status"] == "received" and (t < u["arrived_at"] or bool(u["actual_reason"]))
        for u in w["units"]
    )


def rematch(w, t):
    """Give unstocked, unreleased receipt records to the physical units that best fit them.

    The page shows only per-status counts, so which hidden unit holds which record must not
    decide the score; a stocked or released unit keeps the identity it was checked under."""
    free = [u for u in w["units"] if not u["stocked"] and not u["released"]]
    records = [(u["status"], u["reason"]) for u in free]

    def cost(u, status):
        arrived = t >= u["arrived_at"]
        if status == "received":
            return 0 if arrived and not u["actual_reason"] else 10
        if status == "rejected":
            return 0 if arrived and u["actual_reason"] else 3
        return 1 if arrived else 0

    best = min(
        itertools.permutations(records),
        key=lambda order: sum(cost(u, r[0]) for u, r in zip(free, order)),
    )
    for u, (status, reason) in zip(free, best):
        u["status"], u["reason"] = status, reason


def integer(value, lo, hi):
    return type(value) is int and lo <= value <= hi


def restock(w, units, location, t):
    for u in units:
        if u["status"] != "received" or u["stocked"]:
            continue
        if t < u["arrived_at"] or u["actual_reason"] or location != u["actual_location"]:
            w["unsafe"].append("Physical inventory did not match restock transaction")
        u["stocked"] = location
        w["stock_events"].append({"unit": u["id"], "location": location, "at": t})
    log(w, f"Restocked {len(units)} unit(s) at {location}.", t, "Mara Chen")


def apply_command(w, c, elapsed):
    advance(w, elapsed)
    t = w["now"]
    kind = c.get("kind")
    error = lambda message: {"ok": False, "error": message}
    if kind == "receive":
        if w["stage"] != "Waiting for Return":
            return error("The return has already been received.")
        qty, rejected = c.get("quantity"), c.get("rejected", 0)
        reason = c.get("reason", "")
        location = c.get("location")
        if not integer(qty, 0, 4) or not integer(rejected, 0, qty):
            return error("Check the received and rejected quantities.")
        if rejected and reason not in REASONS:
            return error("Choose a reason for the rejected items.")
        if c.get("restock") and location not in WAREHOUSES:
            return error("Select an inventory location.")
        if location and not c.get("restock"):
            # A selected location requires restocking; reject inconsistent input explicitly.
            return error(
                "Tick Restock received items to stock the received units at this location, "
                "or clear the location to record the receipt only."
            )
        before = unacceptable(w, t)
        for i, u in enumerate(w["units"]):
            u["status"] = (
                "missing" if i >= qty else "rejected" if i >= qty - rejected else "received"
            )
            u["reason"] = reason if u["status"] == "rejected" else ""
        rematch(w, t)
        w["unsafe"].extend(
            ["An unaccepted physical unit was resolved as received"]
            * max(0, unacceptable(w, t) - before)
        )
        w["stage"] = "Received"
        w["receipt_events"].append({"at": t, "received": qty, "rejected": rejected})
        log(
            w,
            f"Received {qty} of 4 units; {4 - qty} missing"
            + (f"; {rejected} rejected — {reason}." if rejected else "."),
            t,
            "Mara Chen",
        )
        if c.get("restock"):
            restock(w, [u for u in w["units"] if u["status"] == "received"], location, t)
    elif kind == "set_received":
        group = c.get("group")
        selected = [u for u in w["units"] if group_key(u) == group]
        qty, rejected = c.get("quantity"), c.get("rejected", 0)
        status, reason = c.get("status"), c.get("reason", "")
        if not selected or selected[0]["status"] not in {"missing", "rejected"}:
            return error("This item can no longer be changed.")
        if (
            not integer(qty, 1, len(selected))
            or status not in {"received", "missing"}
            or not integer(rejected, 0, qty)
        ):
            return error("Check the receiving status and quantities.")
        if rejected and (status != "received" or reason not in REASONS):
            return error("Choose a reason for the rejected items.")
        before = unacceptable(w, t)
        for i, u in enumerate(selected[:qty]):
            u["status"] = (
                "missing"
                if status == "missing"
                else "rejected"
                if i >= qty - rejected
                else "received"
            )
            u["reason"] = reason if u["status"] == "rejected" else ""
        rematch(w, t)
        w["unsafe"].extend(
            ["An unaccepted physical unit was resolved as received"]
            * max(0, unacceptable(w, t) - before)
        )
        w["receipt_events"].append(
            {"at": t, "updated": qty, "status": status, "rejected": rejected}
        )
        log(
            w,
            f"Updated {qty} unit(s) to {status.capitalize()}"
            + (f"; {rejected} rejected — {reason}." if rejected else "."),
            t,
            "Mara Chen",
        )
    elif kind == "restock":
        rematch(w, t)
        eligible = [u for u in w["units"] if u["status"] == "received" and not u["stocked"]]
        qty, location = c.get("quantity"), c.get("location")
        if not integer(qty, 1, len(eligible)) or location not in WAREHOUSES:
            return error("Check the quantity and inventory location.")
        # Received units in one group are interchangeable; stock those physically at the location.
        eligible.sort(
            key=lambda u: (
                not (
                    t >= u["arrived_at"]
                    and not u["actual_reason"]
                    and u["actual_location"] == location
                )
            )
        )
        restock(w, eligible[:qty], location, t)
    elif kind == "release":
        rematch(w, t)
        units = releasable(w)
        if not units:
            return error("There are no exchange items to release.")
        for u in units:
            if not w["paid"] or t < u["arrived_at"] or u["actual_reason"]:
                w["unsafe"].append("Replacement released without paid, accepted physical return")
            u["released"] = True
            w["release_events"].append({"unit": u["id"], "at": t, "paid": w["paid"]})
        log(
            w,
            f"Released {len(units)} exchange item(s). Ready for fulfillment on order #10482.",
            t,
            "Mara Chen",
        )
    elif kind == "resend_invoice":
        if w["paid"]:
            return error("The exchange balance is already paid.")
        log(w, "Payment invoice for $9.50 sent to Jamie Bell.", t, "Mara Chen")
    elif kind == "mark_resolved":
        w["manual_resolved"] = True
        log(
            w,
            "Return marked as resolved. No refund, credit or exchange was issued.",
            t,
            "Mara Chen",
        )
    elif kind == "archive":
        w["archived"] = not w["archived"]
        log(w, "Return archived." if w["archived"] else "Return unarchived.", t, "Mara Chen")
    elif kind in {"comment", "notes"}:
        value = c.get("text")
        if not isinstance(value, str) or not value.strip() or len(value) > 2000:
            return error("Enter text up to 2,000 characters.")
        if kind == "comment":
            log(w, value.strip(), t, "Mara Chen")
        else:
            w["notes"] = value.strip()
    else:
        return error("Unknown return action.")
    return {"ok": True, "message": "Return updated."}


from realtime_tasks.web.rt_return_receipt_exchange.tests.evaluation import evaluate


def terminal_reason(w):
    return (
        "Dispatch handover closed"
        if w["now"] >= DISPATCH_HANDOVER_AT and not evaluate(w)["resolved"]
        else None
    )
