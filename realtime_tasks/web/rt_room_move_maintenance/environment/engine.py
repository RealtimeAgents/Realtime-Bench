"""Room-calendar relocation, historical stay split, and changing maintenance inventory."""

from copy import deepcopy

from datetime import date

import hashlib

import random

TASK_ID = "rt_room_move_maintenance"

STALE_MESSAGE = "The calendar has changed. Review the current room information and try again."

TODAY = "2026-09-17"

PLUMBER_AT = 80

REVISION_AT = 165

ONLINE_SYNC_AT = 230

CLOCK_FIELDS = ("clock_seconds", "plumber_in_seconds", "online_sync_in_seconds")

READY_ROOM = "205"

ASSESSED_END = "2026-09-21"

REVISED_END = "2026-09-23"

DISPLACED = "6151"

MAINTENANCE = "Rafi · Maintenance"


def _label(iso):
    return date.fromisoformat(iso).strftime("%a %d %b %Y")


def create_world(episode_key):
    rng = random.Random(int(hashlib.sha256(episode_key.encode()).hexdigest(), 16))
    ready = READY_ROOM
    rooms = [
        {
            "number": str(n),
            "type": "Double" if n < 208 else "Single",
            "condition": "Clean" if str(n) == ready or n in (201, 202, 207, 208) else "Dirty",
            "attendant": "Sam" if n % 2 else "Jules",
            "comments": [],
            "updated": 0,
        }
        for n in range(201, 209)
    ]
    # Comments are a thread, newest first; the page renders them in that order.
    rooms[2]["comments"] = [
        {
            "at": 0,
            "author": MAINTENANCE,
            "text": (
                "Ceiling leak. Room 203: out of service from September 17 to September 20 for now. "
                "Assessment in progress; I will post the confirmed dates here."
            ),
        }
    ]
    return {
        "rooms": rooms,
        "bookings": [
            {
                "id": "6108",
                "guest": "Maya Chen",
                "status": "In House",
                "type": "Double",
                "adults": 2,
                "arrival": "2026-09-16",
                "departure": "2026-09-20",
                "total": 640,
                "balance": 0,
                "segments": [{"room": "203", "start": "2026-09-16", "end": "2026-09-20"}],
                "notes": [
                    "15:00 · Front desk — Ceiling leak over the bed. Plumber arrives at 15:01:20; Maya and her partner need to be moved before then. Repair updates are in room 203's accommodation comments."
                ],
            },
            {
                "id": "6082",
                "guest": "Rory Ellis",
                "status": "In House",
                "type": "Double",
                "adults": 1,
                "arrival": "2026-09-15",
                "departure": "2026-09-19",
                "total": 640,
                "balance": 0,
                "segments": [{"room": "201", "start": "2026-09-15", "end": "2026-09-19"}],
                "notes": [],
            },
            {
                "id": "6127",
                "guest": "Emilia Soto",
                "status": "Confirmed",
                "type": "Double",
                "adults": 2,
                "arrival": "2026-09-18",
                "departure": "2026-09-22",
                "total": 640,
                "balance": 160,
                "segments": [{"room": "202", "start": "2026-09-18", "end": "2026-09-22"}],
                "notes": [],
            },
            {
                "id": "6135",
                "guest": "Noah Park",
                "status": "Confirmed",
                "type": "Double",
                "adults": 2,
                "arrival": "2026-09-19",
                "departure": "2026-09-21",
                "total": 320,
                "balance": 0,
                "segments": [{"room": "207", "start": "2026-09-19", "end": "2026-09-21"}],
                "notes": [],
            },
            {
                "id": "6142",
                "guest": "Amara Hill",
                "status": "Confirmed",
                "type": "Single",
                "adults": 1,
                "arrival": "2026-09-21",
                "departure": "2026-09-23",
                "total": 220,
                "balance": 220,
                "segments": [{"room": "208", "start": "2026-09-21", "end": "2026-09-23"}],
                "notes": [],
            },
            {
                "id": DISPLACED,
                "guest": "Priya Nair",
                "status": "Confirmed",
                "type": "Double",
                "adults": 2,
                "arrival": "2026-09-21",
                "departure": "2026-09-22",
                "total": 160,
                "balance": 160,
                "segments": [{"room": "203", "start": "2026-09-21", "end": "2026-09-22"}],
                "notes": [],
            },
        ],
        "blocks": [],
        "activity": [],
        "message": "",
        "moved_at": None,
        "elapsed": 0.0,
        "confirmed_end": None,
        "displaced": [],
        "assessment_at": rng.randint(100, 115),
        "assessed": False,
        "revised": False,
        "external_checked": False,
        "violations": [],
        "failed": None,
    }


def room(w, number):
    return next((r for r in w["rooms"] if r["number"] == number), None)


def booking(w, identifier):
    return next((b for b in w["bookings"] if b["id"] == identifier), None)


def overlap(a, b, c, d):
    return a < d and c < b


def occupied(w, number, start, end, except_booking=None, except_block=None):
    return any(
        b["id"] != except_booking
        and b["status"] != "Checked Out"
        and any(
            s["room"] == number and overlap(start, end, s["start"], s["end"]) for s in b["segments"]
        )
        for b in w["bookings"]
    ) or any(
        x["id"] != except_block
        and x["room"] == number
        and overlap(start, end, x["start"], x["end"])
        for x in w["blocks"]
    )


def _log(w, elapsed, text):
    w["activity"].append({"at": round(elapsed, 2), "text": text})
    w["message"] = text


def _confirm(w, at, end, text):
    """Maintenance posts a confirmed closure; reservations it now covers may be re-accommodated."""
    unit = room(w, "203")
    unit["comments"].insert(0, {"at": at, "author": MAINTENANCE, "text": text})
    unit["updated"] = at
    w["confirmed_end"] = end
    for b in w["bookings"]:
        if (
            b["id"] != "6108"
            and b["id"] not in w["displaced"]
            and any(
                s["room"] == "203" and overlap(TODAY, end, s["start"], s["end"])
                for s in b["segments"]
            )
        ):
            w["displaced"].append(b["id"])


def advance(w, elapsed):
    w["elapsed"] = elapsed
    if w["failed"] or w["violations"] or evaluate(w)["resolved"]:
        return
    if elapsed >= PLUMBER_AT and w["moved_at"] is None:
        w["failed"] = "relocation_missed"
        booking(w, "6108")["notes"].append(
            "15:01:20 · Rafi — Crew arrived but room 203 is still occupied. Work cannot begin."
        )
        _log(w, PLUMBER_AT, "Maintenance could not enter room 203.")
        return
    if not w["assessed"] and elapsed >= w["assessment_at"]:
        w["assessed"] = True
        _confirm(
            w,
            w["assessment_at"],
            ASSESSED_END,
            (
                "Assessment complete. Ceiling must dry before repair. Confirmed out-of-service dates: "
                f"September 17 to September {ASSESSED_END[-2:]}. The earlier September 20 estimate is withdrawn. "
                "Reopen for arrivals on the end date."
            ),
        )
        _log(
            w,
            w["assessment_at"],
            "Rafi · Maintenance posted a new accommodation comment on room 203.",
        )
    if not w["revised"] and elapsed >= REVISION_AT:
        w["revised"] = True
        _confirm(
            w,
            REVISION_AT,
            REVISED_END,
            (
                "Joist check done: the void above the ceiling is still wet and needs two more drying days. "
                f"Confirmed out-of-service dates: September 17 to September {REVISED_END[-2:]}. "
                f"The September {ASSESSED_END[-2:]} reopening is withdrawn. Reopen for arrivals on the end date."
            ),
        )
        _log(w, REVISION_AT, "Rafi · Maintenance posted a new accommodation comment on room 203.")
    if elapsed >= ONLINE_SYNC_AT and not w["external_checked"]:
        w["external_checked"] = True
        # The channel sync sells the first night of 203 that is neither occupied nor blocked,
        # exactly as the live strip on the page states.
        for day in range(int(TODAY[-2:]), int(w["confirmed_end"][-2:])):
            start, end = f"2026-09-{day:02d}", f"2026-09-{day + 1:02d}"
            if not occupied(w, "203", start, end):
                w["bookings"].append(
                    {
                        "id": "6160",
                        "guest": "Leo James",
                        "status": "Confirmed",
                        "type": "Double",
                        "adults": 2,
                        "arrival": start,
                        "departure": end,
                        "total": 160,
                        "balance": 0,
                        "notes": [],
                        "segments": [{"room": "203", "start": start, "end": end}],
                    }
                )
                _log(
                    w,
                    ONLINE_SYNC_AT,
                    f"New online reservation · Leo James · room 203 · {start[5:]} to {end[5:]}",
                )
                w["failed"] = "damaged_inventory_sold"
                break


def public_world(w):
    data = {k: deepcopy(w[k]) for k in ("rooms", "bookings", "blocks", "activity", "message")}
    for r in data["rooms"]:
        r["occupancy"] = (
            "Occupied"
            if any(
                b["status"] == "In House"
                and any(
                    s["room"] == r["number"] and s["start"] <= TODAY < s["end"]
                    for s in b["segments"]
                )
                for b in data["bookings"]
            )
            else "Vacant"
        )
    clock = int(w["elapsed"])
    data["clock_seconds"] = clock
    data["plumber_in_seconds"] = max(0, PLUMBER_AT - clock)
    data["online_sync_in_seconds"] = max(0, ONLINE_SYNC_AT - clock)
    return data


def apply_command(w, command, elapsed):
    operation = command.get("operation")
    if operation == "move":
        b = booking(w, command.get("booking"))
        destination = room(w, command.get("destination"))
        if not b or not destination or b["status"] == "Checked Out":
            return {"ok": False, "error": "Reservation or accommodation is unavailable."}
        start = max(TODAY, b["arrival"])
        current = next(s for s in b["segments"] if s["end"] > TODAY)
        if destination["number"] == current["room"]:
            return {"ok": False, "error": "The reservation is already in this accommodation."}
        if destination["type"] != b["type"]:
            return {"ok": False, "error": "Select an accommodation in the booked room type."}
        if occupied(w, destination["number"], start, b["departure"], except_booking=b["id"]):
            return {"ok": False, "error": "Accommodation is unavailable for these dates."}
        # Only Maya and a reservation the confirmed closure has displaced from 203 may be moved.
        if b["id"] != "6108" and b["id"] not in w["displaced"]:
            w["violations"].append("unrelated_guest_moved")
        # Today's housekeeping status matters to a guest moving in today, not to a future arrival.
        if b["status"] == "In House" and destination["condition"] not in ("Clean", "Inspected"):
            w["violations"].append("guest_moved_to_unclean_room")
        previous = current["room"]
        if current["start"] < start:
            current["end"] = start
            b["segments"].append(
                {"room": destination["number"], "start": start, "end": b["departure"]}
            )
        else:
            current["room"] = destination["number"]
        if b["id"] == "6108":
            w["moved_at"] = elapsed if w["moved_at"] is None else w["moved_at"]
        stamp = f"15:{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        b["notes"].append(
            f"{stamp} · Front desk — Room move {previous} → {destination['number']}. Stay dates and rate unchanged."
        )
        _log(w, elapsed, f"{b['guest']} moved to room {destination['number']}.")
    elif operation == "block_save":
        number, start, end, reason = (command.get(k) for k in ("room", "start", "end", "reason"))
        try:
            start, end = date.fromisoformat(start).isoformat(), date.fromisoformat(end).isoformat()
        except (ValueError, TypeError):
            return {
                "ok": False,
                "error": "Choose the From date and the To date (the morning the room reopens).",
            }
        if start < TODAY or start >= end:
            return {"ok": False, "error": "Check the date range. The start cannot be in the past."}
        if not room(w, number) or not isinstance(reason, str) or not reason.strip():
            return {"ok": False, "error": "Select an accommodation and enter a reason."}
        identifier = command.get("block_id")
        existing = next((x for x in w["blocks"] if x["id"] == identifier), None)
        if identifier and not existing:
            return {"ok": False, "error": "This block is no longer available."}
        others = [x for x in w["blocks"] if x["room"] == "203" and x["id"] != identifier]
        if number == "203" and start != TODAY and start not in {x["end"] for x in others}:
            # Require continuous closure coverage and report gaps before saving.
            if others:
                return {
                    "ok": False,
                    "error": (
                        f"Leave no gap in the closure of room 203: start this block on "
                        f"{_label(max(x['end'] for x in others))}, where the current block ends, or edit that block."
                    ),
                }
            return {
                "ok": False,
                "error": f"Room 203 is closed from today. Start its Out of Service block on {_label(TODAY)}.",
            }
        if occupied(w, number, start, end, except_block=identifier):
            if number == "203" and any(overlap(start, end, x["start"], x["end"]) for x in others):
                return {
                    "ok": False,
                    "error": "Room 203 already has an Out of Service block on some of these dates. Click that block on the calendar to edit it.",
                }
            clash = next(
                (
                    (b, s)
                    for b in w["bookings"]
                    if b["status"] != "Checked Out"
                    for s in b["segments"]
                    if s["room"] == number and overlap(start, end, s["start"], s["end"])
                ),
                None,
            )
            if clash:
                b, s = clash
                return {
                    "ok": False,
                    "error": (
                        f"Accommodation is unavailable for these dates: {b['guest']} ({b['id']}) "
                        f"is assigned to room {number} from {_label(s['start'])}."
                    ),
                }
            return {"ok": False, "error": "Accommodation is unavailable for these dates."}
        if number != "203":
            w["violations"].append("unrelated_room_blocked")
        values = {"room": number, "start": start, "end": end, "reason": reason.strip()[:1000]}
        if existing:
            existing.update(values)
        else:
            w["blocks"].append({"id": f"block-{len(w['blocks']) + 1}", **values})
        _log(w, elapsed, f"Out of service saved · room {number} · {start[5:]} to {end[5:]}")
    elif operation == "note":
        b = booking(w, command.get("booking"))
        note = command.get("text", "")
        if not b or not isinstance(note, str) or not note.strip():
            return {"ok": False, "error": "Enter a reservation note."}
        if b["id"] != "6108":
            w["violations"].append("unrelated_guest_note_changed")
        b["notes"].append("Front desk — " + note.strip()[:1000])
        _log(w, elapsed, "Reservation note saved.")
    else:
        return {"ok": False, "error": "Unknown calendar operation."}
    return {"ok": True, "message": w["message"]}


from realtime_tasks.web.rt_room_move_maintenance.tests.evaluation import evaluate


def terminal_reason(w):
    return w["failed"] or ("unsafe_calendar_change" if w["violations"] else None)
