"""Bespoke coordinator/driver conversation with recipient-specific handoff."""

import hashlib

import math

import random

from copy import deepcopy

TASK_ID = "rt_pickup_driver_handoff"

CLOCK_FIELDS = ("clock_seconds", "minutes", "reply_due_s", "arrival_in_s")

STALE_MESSAGE = "The ride changed. Review its current details and try again."

DOORS = ("Cedar Door", "Gallery Door", "Harbor Door")

REPLY_WINDOW = 90

DRIVERS = {
    "sam": {
        "id": "sam",
        "name": "Sam Rivera",
        "phone": "(415) 555-0148",
        "car": "Silver sedan",
        "plate": "7KLM204",
    },
    "imani": {
        "id": "imani",
        "name": "Imani Brooks",
        "phone": "(415) 555-0173",
        "car": "Blue hatchback",
        "plate": "8PQR516",
    },
}


def create_world(seed):
    rng = random.Random(int(hashlib.sha256(str(seed).encode()).hexdigest()[:16], 16))
    first, second, other = rng.sample(list(DOORS), 3)

    def entrances(good, stairs, far):
        rows = [
            dict(name=good, metres=rng.randint(35, 58), access="step-free"),
            dict(name=stairs, metres=rng.randint(18, 32), access="steps"),
            dict(name=far, metres=rng.randint(75, 110), access="step-free"),
        ]
        rng.shuffle(rows)
        return rows

    before = entrances(first, second, other)
    after = entrances(second, other, first)
    w = dict(
        elapsed=0,
        swap_at=rng.randint(108, 122),
        arrival_at=rng.randint(245, 260),
        read_delay=rng.randint(5, 8),
        swapped=False,
        report_at=None,
        reported=False,
        before=before,
        after=after,
        first_door=first,
        second_door=second,
        driver="sam",
        ride_status="Getting picked up",
        cancelled=False,
        additional_rides=[],
        messages=[],
        next_id=1,
        pending=[],
        instructions={},
        acknowledgements={},
        first_met=None,
        boarded=False,
        ended=False,
        final_snapshot=None,
        notes="Blue jacket; waiting inside the conference foyer.",
    )
    _add(w, "system", "sam", "Alex started a conversation on behalf of Lena.", -55)
    _add(
        w,
        "coordinator",
        "sam",
        "Hi Sam, I am coordinating Lena's pickup. What are the entrances like today?",
        -50,
        "Read",
    )
    _add(w, "driver", "sam", _report(before, False), -15)
    return w


def _report(rows, changed):
    lead = (
        "I've checked the current access at the centre. The temporary barriers have moved: "
        if changed
        else "I'm approaching the conference centre. From Lena's waiting place: "
    )
    detail = "; ".join(
        f"{r['name']} is {r['metres']} m, {'with steps' if r['access'] == 'steps' else 'step-free'}"
        for r in rows
    )
    return lead + detail + ". Which entrance should I head for?"


def _add(w, who, driver, text, at, status="", door=None):
    # `door` is set only on structured meeting-point messages; chat wording is never interpreted.
    msg = {
        "id": w["next_id"],
        "who": who,
        "driver": driver,
        "text": text,
        "at": round(at, 2),
        "status": status,
        "door": door,
    }
    w["next_id"] += 1
    w["messages"].append(msg)
    return msg


def advance(w, elapsed):
    if w["ended"]:
        return
    until = max(w["elapsed"], elapsed)
    events = []
    if w["first_met"] is None and until >= REPLY_WINDOW:
        events.append((REPLY_WINDOW, "deadline", None))
    if not w["swapped"] and until >= w["swap_at"]:
        events.append((w["swap_at"], "swap", None))
    if w["report_at"] is not None and not w["reported"] and until >= w["report_at"]:
        events.append((w["report_at"], "report", None))
    for job in w["pending"]:
        if job["at"] <= until:
            events.append((job["at"], "read", job))
    if w["final_snapshot"] is None and until >= w["arrival_at"]:
        events.append((w["arrival_at"], "arrival", None))
    for at, kind, job in sorted(events, key=lambda row: row[0]):
        if kind == "deadline":
            latest = w["instructions"].get("sam")
            w["first_met"] = bool(
                latest and latest["door"] == w["first_door"] and latest["at"] <= REPLY_WINDOW
            )
        elif kind == "swap" and not w["cancelled"]:
            w["swapped"] = True
            w["driver"] = "imani"
            _add(
                w,
                "system",
                "imani",
                "Your driver changed from Sam Rivera to Imani Brooks. "
                "Imani has not received or acknowledged a meeting point.",
                at,
            )
        elif kind == "report" and not w["cancelled"]:
            w["reported"] = True
            _add(w, "driver", "imani", _report(w["after"], True), at)
        elif kind == "read":
            w["pending"].remove(job)
            msg = next(m for m in w["messages"] if m["id"] == job["message_id"])
            if w["cancelled"] or job["driver"] != w["driver"]:
                continue
            msg["status"] = "Read"
            if job["driver"] == "imani" and not job["after_report"]:
                continue  # Imani's entrance report is her reply to anything sent before it.
            if job["door"]:
                w["acknowledgements"][job["driver"]] = {
                    "message_id": msg["id"],
                    "door": job["door"],
                    "at": at,
                }
                _add(
                    w, "driver", job["driver"], f"Understood. I'll meet Lena at {job['door']}.", at
                )
            else:
                _add(
                    w,
                    "driver",
                    job["driver"],
                    "Thanks. Please send the entrance you would like me to use.",
                    at,
                )
        elif kind == "arrival":
            current = w["instructions"].get("imani")
            ack = w["acknowledgements"].get("imani")
            ready = bool(
                current
                and current["after_report"]
                and ack
                and ack["message_id"] == current["message_id"]
                and current["door"] == w["second_door"]
                and ack["at"] < at
            )
            w["final_snapshot"] = dict(ready=ready, first_met=bool(w["first_met"]))
            if not w["cancelled"]:
                w["ride_status"] = "Driver arrived"
                _add(
                    w, "system", w["driver"], "Your driver has arrived at the pickup location.", at
                )
    w["elapsed"] = until
    if w["cancelled"]:
        w["ended"] = True
    elif w["final_snapshot"] and until >= w["arrival_at"] + 8 and w["final_snapshot"]["ready"]:
        w["ride_status"] = "Ride in progress"
        w["boarded"] = True
        w["ended"] = True
        _add(w, "system", w["driver"], "Lena's ride has started.", w["arrival_at"] + 8)
    elif until >= w["arrival_at"] + 35:
        w["ride_status"] = "Canceled — rider not picked up"
        w["ended"] = True
        _add(w, "system", w["driver"], "This pickup could not be completed.", w["arrival_at"] + 35)


def _meeting(w):
    """The current driver's meeting point and whether they acknowledged that exact message."""
    current = w["instructions"].get(w["driver"])
    ack = w["acknowledgements"].get(w["driver"])
    acked = bool(current and ack and ack["message_id"] == current["message_id"])
    return {
        "door": current["door"] if current else None,
        "sent_at": current["at"] if current else None,
        "acknowledged": acked,
        "acknowledged_at": ack["at"] if acked else None,
    }


def public_world(w):
    elapsed = w["elapsed"]
    first_reply = w["instructions"].get("sam")
    return {
        "ride_id": "WF-7308",
        "rider": "Lena Ortiz",
        "phone": "(415) 555-0116",
        "ride_type": "Standard · 4 seats",
        "pickup": "Harbor Conference Centre, 210 Bay Street",
        "destination": "Juniper Hotel, 84 Market Street",
        "coordinator": "Alex Morgan",
        "program": "Harbor Events",
        "requested": "17 Sep 2026, 3:54 PM",
        "driver": dict(DRIVERS[w["driver"]]),
        "status": w["ride_status"],
        "minutes": max(0, int((w["arrival_at"] - elapsed + 59) // 60)),
        "clock_seconds": 16 * 3600 + int(elapsed),
        "messages": deepcopy(w["messages"]),
        "notes": w["notes"],
        "cancelled": w["cancelled"],
        "additional_rides": deepcopy(w["additional_rides"]),
        "entrances": list(DOORS),
        "meeting": _meeting(w),
        "swapped": w["swapped"],
        "first_reply_at": first_reply["at"] if first_reply else None,
        "reply_due_s": None
        if first_reply or elapsed >= REPLY_WINDOW
        else math.ceil(REPLY_WINDOW - elapsed),
        "arrival_in_s": max(0, math.ceil(w["arrival_at"] - elapsed)),
    }


def _deliver(w, text, door, elapsed):
    """Post a coordinator message to the current driver; schedule their read and Imani's first report."""
    driver = w["driver"]
    msg = _add(w, "coordinator", driver, text, elapsed, "Sent", door)
    w["pending"].append(
        {
            "at": elapsed + w["read_delay"],
            "message_id": msg["id"],
            "driver": driver,
            "door": door,
            "after_report": w["reported"],
        }
    )
    if driver == "imani" and w["report_at"] is None:
        w["report_at"] = elapsed + w["read_delay"] + 3
    return msg


def apply_command(w, command, elapsed):
    kind = command.get("type")
    if kind == "send":
        text = command.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 1200:
            return dict(ok=False, error="Enter a message of up to 1,200 characters.")
        if w["ride_status"] != "Getting picked up":
            return dict(ok=False, error="Messaging is unavailable for this ride.")
        _deliver(w, text.strip(), None, elapsed)
        return dict(ok=True, message="Message sent.")
    if kind == "meeting_point":
        door, note = command.get("door"), command.get("note") or ""
        if door not in DOORS:
            return {"ok": False, "error": "Choose an entrance first."}
        if not isinstance(note, str) or len(note) > 300:
            return {"ok": False, "error": "Keep the note to 300 characters."}
        if w["ride_status"] != "Getting picked up":
            return {"ok": False, "error": "Messaging is unavailable for this ride."}
        text = f"Meeting point: {door}." + (f" {note.strip()}" if note.strip() else "")
        msg = _deliver(w, text, door, elapsed)
        w["instructions"][w["driver"]] = {
            "door": door,
            "at": elapsed,
            "message_id": msg["id"],
            "after_report": w["reported"],
        }
        return {"ok": True, "message": f"Meeting point sent to {DRIVERS[w['driver']]['name']}."}
    if kind == "cancel":
        w["cancelled"] = True
        w["ride_status"] = "Canceled"
        _add(w, "system", w["driver"], "Alex canceled this ride.", elapsed)
        w["ended"] = True
        return dict(ok=True, message="Ride canceled.")
    if kind == "request":
        fields = ("rider", "phone", "pickup", "destination")
        if any(not isinstance(command.get(f), str) or not command[f].strip() for f in fields):
            return dict(ok=False, error="Complete the rider, mobile number and both addresses.")
        if command.get("ride_type") not in ("Standard", "XL", "Comfort"):
            return dict(ok=False, error="Choose a ride type.")
        w["additional_rides"].append(
            {
                **{f: command[f].strip() for f in fields},
                "id": f"WF-{7310 + len(w['additional_rides'])}",
                "status": "Looking for driver",
                "ride_type": command["ride_type"],
            }
        )
        return dict(ok=True, message="Ride requested. Looking for a driver.")
    return dict(ok=False, error="This action is unavailable.")


from realtime_tasks.web.rt_pickup_driver_handoff.tests.evaluation import evaluate


def terminal_reason(w):
    return "Ride finished" if w["ended"] and not evaluate(w)["resolved"] else None
