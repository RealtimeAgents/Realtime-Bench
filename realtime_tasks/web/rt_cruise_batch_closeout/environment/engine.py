"""Bespoke supplier Agenda: independent sales, settlements and notifications."""

from copy import deepcopy

import hashlib

import random

TASK_ID = "rt_cruise_batch_closeout"

CLOCK_FIELDS = ("now", "sales_stop_in_s", "handoff_in_s")

STALE_MESSAGE = "These bookings changed. Refresh the data and review your selection."

STATUSES = ("Cancelled", "Confirmed", "Pending customer", "Pending supplier", "On Hold")

EMAILS = ("Order Details", "Order Cancellation", "Gift Cards")

SALES_STOP_AT = 60

HANDOFF_AT = 285

CALLBACK_AT = (100, 125)

REFUND_PROCESSING_S = 18

EMAIL_DELIVERY_S = 9


def create_world(seed):
    r = random.Random(int(hashlib.sha256(str(seed).encode()).hexdigest()[:16], 16))
    sessions = [
        dict(
            id="s-41",
            date="2026-09-18",
            start="13:00",
            end="14:30",
            product="Headland cruise",
            capacity=18,
        ),
        dict(
            id="s-42",
            date="2026-09-18",
            start="15:00",
            end="16:30",
            product="Headland cruise",
            capacity=18,
        ),
        dict(
            id="s-63",
            date="2026-09-18",
            start="13:30",
            end="14:30",
            product="Harbour loop",
            capacity=16,
        ),
        dict(
            id="s-71",
            date="2026-09-19",
            start="13:00",
            end="14:30",
            product="Headland cruise",
            capacity=18,
        ),
    ]
    people = r.sample(["Maya Brooks", "Ellis Grant", "Robin Shah", "Alex Morgan"], 4)
    codes = r.sample(range(3101, 3900), 4)
    orders = []
    for i, s in enumerate(sessions):
        qty = r.choice([2, 3, 4])
        o = dict(
            id="MC" + str(codes[i]),
            session=s["id"],
            customer=people[i],
            email=people[i].lower().replace(" ", ".") + "@example.net",
            phone="+1 202 555 01" + str(20 + i),
            quantity=qty,
            unit_price=4500 if i != 2 else 3000,
            status="Confirmed",
            notes="No special requirements.",
            activity=[],
            refunds=[],
            vouchers=[],
            emails=[],
        )
        o["paid"] = qty * o["unit_price"]
        o["payment"] = "PAY-" + str(r.randint(100000, 999999))
        o["last4"] = str(r.randint(1000, 9999))
        if i < 2:
            o["notes"] = (
                "Nora, 11:58: I reached the customer about the weather cancellation. They are discussing a card refund or returning on a gift voucher. I will record their decision after the call."
            )
        o["activity"] = [
            dict(at=-600, author="Online booking", text="Order confirmed. Card payment received.")
        ]
        orders.append(o)
        s.update(
            available=s["capacity"] - qty,
            deleted=False,
            notes="Coastal operations · Direct bookings",
        )
    # Fixed branch: the 13:00 customer wants a card refund, the 15:00 customer wants vouchers.
    # Only names, order codes and quantities vary with the seed.
    outcomes = ["refund", "voucher"]
    events = [
        (SALES_STOP_AT, "sales-check", None),
        (70, "online-demand", "s-41"),
        (CALLBACK_AT[0], "response", orders[0]["id"]),
        (CALLBACK_AT[1], "response", orders[1]["id"]),
        (265, "online-demand", "s-42"),
    ]
    return dict(
        now=0,
        company="Morrow Coast Tours",
        user="Sam Reed",
        role="Booking Manager",
        sessions=sessions,
        orders=orders,
        _events=sorted(events),
        _done=[],
        _choices={orders[i]["id"]: outcomes[i] for i in range(2)},
        _affected=[o["id"] for o in orders[:2]],
        _first_closure_ok=False,
        _new_cancelled_sales=[],
        _protected=deepcopy(orders[2:]),
        _protected_sessions=deepcopy(sessions[2:]),
        _sequence=0,
    )


def order(w, oid):
    return next((o for o in w["orders"] if o["id"] == oid), None)


def session(w, sid):
    return next((s for s in w["sessions"] if s["id"] == sid), None)


def log(o, at, text, author="Sam Reed"):
    o["activity"].append(dict(at=at, author=author, text=text))


def recalculate(w, sid):
    s = session(w, sid)
    occupied = sum(
        o["quantity"] for o in w["orders"] if o["session"] == sid and o["status"] != "Cancelled"
    )
    s["available"] = max(0, s["capacity"] - occupied)


def send_email(w, o, kind, at):
    w["_sequence"] += 1
    codes = [v["code"] for v in o["vouchers"]]
    s = session(w, o["session"])
    o["emails"].append(
        dict(
            id="mail-" + str(w["_sequence"]),
            at=at,
            kind=kind,
            to=o["email"],
            status="Sending",
            status_snapshot=o["status"],
            product=s["product"],
            date=s["date"],
            start=s["start"],
            quantity=o["quantity"],
            codes=codes if kind == "Gift Cards" else [],
        )
    )
    log(o, at, kind + " email sent.")


def refund(w, o, amount, at):
    if amount:
        w["_sequence"] += 1
        o["refunds"].append(
            dict(id="RF-" + str(w["_sequence"]), amount=amount, at=at, status="Pending")
        )
        log(o, at, "Card refund submitted.")


def advance(w, elapsed):
    for at, event, target in w["_events"]:
        key = (at, event, target)
        if key in w["_done"] or at > elapsed:
            continue
        w["_done"].append(key)
        if event == "sales-check":
            w["_first_closure_ok"] = all(
                session(w, sid)["available"] == 0 for sid in ("s-41", "s-42")
            )
        elif event == "response":
            o = order(w, target)
            message = (
                "Please return the full payment to my card; I cannot come another day."
                if w["_choices"][target] == "refund"
                else "We would like gift vouchers for the same number of Headland cruise places, to choose a new date later. Please keep the payment."
            )
            o["notes"] = "Nora, customer callback: “" + message + "”"
            log(o, at, o["notes"], "Nora Bell")
            recalculate(w, o["session"])
        elif event == "online-demand":
            s = session(w, target)
            if s["available"] > 0 and not s["deleted"]:
                sid = "MC9" + str(int(at))
                new = dict(
                    id=sid,
                    session=target,
                    customer="Online guest",
                    email="online.guest@example.net",
                    phone="+1 202 555 0198",
                    quantity=1,
                    unit_price=4500,
                    paid=4500,
                    payment="PAY-" + sid,
                    last4="6912",
                    status="Confirmed",
                    notes="Online booking.",
                    activity=[],
                    refunds=[],
                    vouchers=[],
                    emails=[],
                )
                log(new, at, "Order confirmed. Card payment received.", "Online booking")
                w["orders"].append(new)
                w["_new_cancelled_sales"].append(sid)
                recalculate(w, target)
    for o in w["orders"]:
        for row in o["refunds"]:
            if row["status"] == "Pending" and elapsed >= row["at"] + REFUND_PROCESSING_S:
                row["status"] = "Processed"
        for row in o["emails"]:
            if row["status"] == "Sending" and elapsed >= row["at"] + EMAIL_DELIVERY_S:
                row["status"] = "Delivered"
    w["now"] = elapsed


def public_world(w):
    public = {k: deepcopy(w[k]) for k in ("now", "company", "user", "role", "sessions", "orders")}
    public["sales_stop_in_s"] = max(0, SALES_STOP_AT - int(w["now"]))
    public["handoff_in_s"] = max(0, HANDOFF_AT - int(w["now"]))
    return public


def apply_command(w, c, elapsed):
    action = c.get("action")
    if action in ("close_sessions", "delete_sessions"):
        ids = c.get("sessions")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
            return dict(ok=False, error="Select a session.")
        selected = [session(w, sid) for sid in dict.fromkeys(ids)]
        if any(s is None or s["deleted"] for s in selected):
            return dict(ok=False, error="A selected session is no longer available.")
        if action == "delete_sessions" and any(o["session"] in ids for o in w["orders"]):
            return dict(ok=False, error="Sessions with attached orders cannot be deleted.")
        for s in selected:
            s["available"] = 0
            if action == "delete_sessions":
                s["deleted"] = True
        return dict(
            ok=True,
            message=f"{len(selected)} sessions "
            + ("closed." if action == "close_sessions" else "deleted."),
        )
    if action == "edit_availability":
        s = session(w, c.get("session"))
        value = c.get("available")
        if not s or s["deleted"] or type(value) is not int or not 0 <= value <= s["capacity"]:
            return dict(ok=False, error="Enter availability between 0 and the session capacity.")
        s["available"] = value
        return dict(ok=True, message="Session saved.")
    if action == "update_orders":
        ids = c.get("orders")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
            return dict(ok=False, error="Select an order.")
        selected = [order(w, oid) for oid in dict.fromkeys(ids)]
        status, mail = c.get("status"), c.get("email")
        if (
            any(o is None for o in selected)
            or (status and status not in STATUSES)
            or (mail and mail not in EMAILS)
        ):
            return dict(ok=False, error="Review the selected orders and form values.")
        if not (status or mail or c.get("refund") or c.get("vouchers")):
            return dict(ok=False, error="Choose an update to apply.")
        if any(type(c.get(k, False)) is not bool for k in ("refund", "vouchers")):
            return dict(ok=False, error="Invalid update options.")
        premature = (
            mail == "Order Cancellation"
            and (status or "") != "Cancelled"
            and any(o["status"] != "Cancelled" for o in selected)
        )
        if premature:
            # A cancellation email carrying a Confirmed status snapshot fails the notification
            # check for the rest of the episode (Email menu on a not-yet-cancelled order).
            # Refuse it with the reason instead of recording it silently.
            return {
                "ok": False,
                "error": "The Order Cancellation email is available for cancelled orders "
                "only. Set the order status to Cancelled first or in the same update.",
            }
        unbacked = (
            mail == "Gift Cards"
            and not c.get("vouchers")
            and any(not o["vouchers"] for o in selected)
        )
        if unbacked:
            # A Gift Cards email with no voucher codes reads as a finished voucher settlement on the
            # page but can never satisfy the notification check (Email menu on an order whose
            # vouchers were never issued). Refuse it and name where vouchers are created.
            return {
                "ok": False,
                "error": "The Gift Cards email needs vouchers first. Issue them with "
                '"Issue vouchers" on the order page or "Create and apply '
                'vouchers" in Update Orders.',
            }
        for o in selected:
            changed = False
            if status:
                changed = True
                o["status"] = status
                log(o, elapsed, "Order status changed to " + status + ".")
            if c.get("refund"):
                changed = True
                remainder = o["paid"] - sum(v["amount"] for v in o["refunds"])
                refund(w, o, max(0, remainder), elapsed)
            if c.get("vouchers"):
                changed = True
                for _ in range(o["quantity"]):
                    w["_sequence"] += 1
                    o["vouchers"].append(
                        dict(
                            code="WC" + str(700000 + w["_sequence"]),
                            quantity=1,
                            product=session(w, o["session"])["product"],
                            at=elapsed,
                        )
                    )
                log(o, elapsed, "Gift vouchers created for the booked quantity.")
                send_email(w, o, "Gift Cards", elapsed)
            if mail:
                send_email(w, o, mail, elapsed)
            if changed:
                recalculate(w, o["session"])
        return dict(ok=True, message=f"{len(selected)} orders updated.")
    if action == "refund_payment":
        o = order(w, c.get("order"))
        amount = c.get("amount")
        if (
            not o
            or type(amount) is not int
            or amount <= 0
            or amount > o["paid"] - sum(v["amount"] for v in o["refunds"])
        ):
            return dict(
                ok=False, error="Enter an amount no greater than the remaining card payment."
            )
        refund(w, o, amount, elapsed)
        recalculate(w, o["session"])
        return dict(ok=True, message="Refund submitted.")
    if action == "log_note":
        o = order(w, c.get("order"))
        text = str(c.get("text", "")).strip()
        if not o or not text or len(text) > 1000:
            return dict(ok=False, error="Enter a note of up to 1000 characters.")
        log(o, elapsed, text)
        recalculate(w, o["session"])
        return dict(ok=True, message="Note saved.")
    return dict(ok=False, error="Action unavailable.")


from realtime_tasks.web.rt_cruise_batch_closeout.tests.evaluation import evaluate


def terminal_reason(w):
    return "The afternoon operations handoff has started." if w["now"] >= HANDOFF_AT else None
