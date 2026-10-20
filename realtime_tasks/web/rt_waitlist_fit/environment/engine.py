"""Appointment waitlist qualification, live client consent and interval allocation."""

import hashlib

from copy import deepcopy

from datetime import datetime

TASK_ID = "rt_waitlist_fit"

CLOCK_FIELDS = ("clock_seconds",)

CLOCK_START = 11 * 3600 + 58 * 60 + 35

MEETING_AT = 85

FIRST_REPLY_AT = 100

OPENING_CLOSES_AT = 250

STALE_MESSAGE = "The calendar or conversation changed. Review the latest details and save again."

SERVICES = {
    "cut": {"name": "Cut & finish", "minutes": 45, "price": 48},
    "trim": {"name": "Express trim", "minutes": 30, "price": 30},
    "colour": {"name": "Colour & finish", "minutes": 90, "price": 95},
}


def _message(w, client, author, text, at, automatic=False):
    thread = w["messages"][client]
    thread.append(
        {
            "id": f"{client}-{len(thread)}",
            "author": author,
            "text": text,
            "at": at,
            "automatic": automatic,
        }
    )


def create_world(seed):
    d = hashlib.sha256(seed.encode()).digest()
    names = ["Rosa Kim", "Avery Lane", "Devon Bell", "Tess Morgan"]
    shift = d[0] % 4
    names = names[shift:] + names[:shift]
    requests = []
    for i, (service, staff, start, end, created) in enumerate(
        [
            ("colour", ["Leah"], 840, 930, "2026-09-14 09:00"),
            ("cut", ["Leah", "Noor"], 840, 915, "2026-09-15 10:15"),
            ("trim", ["Leah"], 840, 900, "2026-09-16 13:20"),
            ("cut", ["Leah", "Noor"], 840, 915, "2026-09-17 08:35"),
        ]
    ):
        requests.append(
            {
                "id": f"c_{d[2 + i : 5 + i].hex()}",
                "name": names[i],
                "service": service,
                "staff": staff,
                "start": start,
                "end": end,
                "created": created,
                "date": "2026-09-17",
                "note": "",
                "removed": False,
                "phone": f"07700 90012{i}",
                "email": names[i].lower().replace(" ", ".") + "@example.net",
            }
        )
    # Only the seed-independent name rotation and ids vary; the branch is always the strict one.
    requests[1]["reachable_until"] = CLOCK_START + MEETING_AT
    appointments = [
        {
            "id": "a11",
            "client_id": None,
            "name": "Elena Park",
            "staff": "Leah",
            "service": "colour",
            "start": 750,
            "end": 840,
        },
        {
            "id": "a12",
            "client_id": None,
            "name": "Priya Shah",
            "staff": "Leah",
            "service": "colour",
            "start": 900,
            "end": 990,
        },
        {
            "id": "a21",
            "client_id": None,
            "name": "Sam Ellis",
            "staff": "Noor",
            "service": "colour",
            "start": 780,
            "end": 870,
        },
        {
            "id": "a22",
            "client_id": None,
            "name": "Taylor Moss",
            "staff": "Noor",
            "service": "colour",
            "start": 930,
            "end": 1020,
        },
    ]
    for a in appointments:
        a.update(
            date="2026-09-17",
            status="Confirmed",
            note="",
            notification="Sent",
            channel="Text",
            original=True,
            notify_at=None,
        )
    w = {
        "requests": requests,
        "appointments": appointments,
        "messages": {r["id"]: [] for r in requests},
        "elapsed": 0,
        "outreach": [],
        "reply_due": {},
        "acknowledged": set(),
        "confirmed": {},
        "first_id": requests[1]["id"],
        "online_booked": False,
        "closed": False,
        "violations": [],
        "counter": 0,
    }
    for r in requests:
        _message(
            w,
            r["id"],
            "client",
            "Please let me know if anything opens up this afternoon. You can text me here.",
            -1200,
        )
    _message(
        w,
        requests[1]["id"],
        "client",
        "I'm checking my phone now, but I'm in a meeting from 12:00. Is there an opening for my cut?",
        -8,
    )
    return w


def _time(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _request(w, client):
    return next((r for r in w["requests"] if r["id"] == client), None)


def _violate(w, reason):
    if reason not in w["violations"]:
        w["violations"].append(reason)


def advance(w, elapsed):
    if elapsed < w["elapsed"]:
        return
    if elapsed >= 65 and not w["online_booked"]:
        w["online_booked"] = True
        w["appointments"].append(
            {
                "id": "a-online",
                "client_id": None,
                "name": "Casey Holt",
                "staff": "Leah",
                "service": "trim",
                "start": 840,
                "end": 870,
                "date": "2026-09-17",
                "status": "Confirmed",
                "note": "Booked online",
                "notification": "Sent",
                "channel": "Text",
                "original": True,
                "notify_at": None,
            }
        )
    if elapsed >= MEETING_AT and not any(c == w["first_id"] for c, _ in w["outreach"]):
        if not w["closed"]:
            _message(
                w,
                w["first_id"],
                "client",
                "I'm going into my meeting now, so I can't arrange a visit today.",
                MEETING_AT,
            )
            w["closed"] = True
        _violate(w, "initial_client_missed")
    for client, contacted in w["outreach"]:
        if elapsed >= contacted + 8 and client not in w["acknowledged"]:
            _message(
                w,
                client,
                "client",
                "Thanks for getting back to me. Let me check my diary.",
                contacted + 8,
            )
            w["acknowledged"].add(client)
        if elapsed >= w["reply_due"][client] and client not in w["confirmed"]:
            request = _request(w, client)
            start, end, staff = request["start"], request["end"], request["staff"][:]
            if client == w["first_id"]:
                # Fixed strict branch: the first client narrows to Leah from 14:30, which no longer fits.
                start, end, staff = 870, 915, ["Leah"]
            w["confirmed"][client] = {
                "start": start,
                "end": end,
                "staff": staff,
                "at": w["reply_due"][client],
            }
            who = "Leah only" if staff == ["Leah"] else "either Leah or Noor"
            _message(
                w,
                client,
                "client",
                f"Please book my {SERVICES[request['service']]['name'].lower()} with {who}. I can arrive from {_time(start)} and need to be finished by {_time(end)} today.",
                w["reply_due"][client],
            )
    for a in w["appointments"]:
        if a["status"] == "Confirmed" and a["notify_at"] is not None and elapsed >= a["notify_at"]:
            a["notification"] = "Sent"
            if a["client_id"] in w["messages"]:
                _message(
                    w,
                    a["client_id"],
                    "salon",
                    f"Appointment confirmed: {SERVICES[a['service']]['name']} with {a['staff']}, 17 Sep at {_time(a['start'])}. Sent by {a['channel'].lower()}.",
                    a["notify_at"],
                    True,
                )
            a["notify_at"] = None
    if elapsed >= OPENING_CLOSES_AT and not evaluate(w)["resolved"]:
        _violate(w, "appointment_opportunity_closed")
    w["elapsed"] = elapsed


def _overlap(w, staff, start, end, exclude=None):
    return [
        a
        for a in w["appointments"]
        if a["id"] != exclude
        and a["status"] == "Confirmed"
        and a["staff"] == staff
        and a["start"] < end
        and start < a["end"]
    ]


def _fits(w, r, preference):
    length = SERVICES[r["service"]]["minutes"]
    return any(
        not _overlap(w, staff, start, start + length)
        for staff in preference["staff"]
        for start in range(
            max(720, preference["start"]), min(1020, preference["end"]) - length + 1, 5
        )
    )


def public_world(w):
    appointments = [
        {k: deepcopy(v) for k, v in a.items() if k not in {"original", "notify_at"}}
        for a in w["appointments"]
    ]
    return {
        "date": "2026-09-17",
        "location": "Willow & Reed Salon",
        "staff": ["Leah", "Noor"],
        "services": deepcopy(SERVICES),
        "requests": deepcopy(w["requests"]),
        "appointments": appointments,
        "messages": deepcopy(w["messages"]),
        "clock_start": CLOCK_START,
        "opening_closes_at": CLOCK_START + OPENING_CLOSES_AT,
        "clock_seconds": CLOCK_START + int(w["elapsed"]),
    }


def apply_command(w, command, elapsed):
    op = command.get("op")
    if op == "message":
        client, text = command.get("client_id"), command.get("text")
        r = _request(w, client)
        if not r or not isinstance(text, str) or not text.strip() or len(text) > 1600:
            return {
                "ok": False,
                "error": "Select a customer and enter a message of up to 1,600 characters.",
            }
        _message(w, client, "salon", text.strip(), elapsed)
        if not any(c == client for c, _ in w["outreach"]):
            if not w["outreach"] and client != w["first_id"]:
                _violate(w, "waitlist_priority_skipped")
            elif w["outreach"]:
                for prior in sorted(w["requests"], key=lambda r: r["created"]):
                    if prior["created"] >= r["created"]:
                        break
                    preference = w["confirmed"].get(prior["id"], prior)
                    if not prior["removed"] and _fits(w, prior, preference):
                        _violate(w, "compatible_earlier_client_skipped")
            w["outreach"].append((client, elapsed))
            w["reply_due"][client] = (
                max(FIRST_REPLY_AT, elapsed + 18) if client == w["first_id"] else elapsed + 18
            )
        return {"ok": True, "message": "Message sent."}
    if op in {"staff_note", "remove_request"}:
        r = _request(w, command.get("client_id"))
        if not r:
            return {"ok": False, "error": "This waitlist request is unavailable."}
        if op == "staff_note":
            note = command.get("note", "")
            if not isinstance(note, str) or len(note) > 1200:
                return {"ok": False, "error": "Use a note of up to 1,200 characters."}
            r["note"] = note
        else:
            if not any(
                a["client_id"] == r["id"] and a["status"] == "Confirmed" for a in w["appointments"]
            ):
                return {
                    "ok": False,
                    "error": "Only clients with a booked appointment can be removed from the waitlist. Keep this request on the list.",
                }
            r["removed"] = True
        return {"ok": True, "message": "Waitlist updated."}
    if op == "send_confirmation":
        a = next((a for a in w["appointments"] if a["id"] == command.get("appointment_id")), None)
        if not a or a["status"] != "Confirmed" or a["client_id"] not in w["messages"]:
            return {"ok": False, "error": "Choose a booked waitlist appointment to confirm."}
        a.update(channel="Text", notification="Queued", notify_at=elapsed + 8)
        return {"ok": True, "message": "Confirmation queued. It is delivered by text in 8 seconds."}
    if op == "cancel_appointment":
        a = next((a for a in w["appointments"] if a["id"] == command.get("appointment_id")), None)
        if not a or a["status"] != "Confirmed":
            return {"ok": False, "error": "The appointment is no longer available."}
        if a["original"]:
            _violate(w, "existing_appointment_changed")
        a["status"] = "Canceled"
        a["notify_at"] = None
        return {"ok": True, "message": "Appointment canceled."}
    if op == "save_appointment":
        existing = next(
            (a for a in w["appointments"] if a["id"] == command.get("appointment_id")), None
        )
        client = command.get("client_id")
        r = _request(w, client)
        service, staff = command.get("service"), command.get("staff")
        try:
            parsed = datetime.strptime(str(command.get("time")), "%H:%M")
            start = parsed.hour * 60 + parsed.minute
        except ValueError:
            return {"ok": False, "error": "Enter a valid appointment time."}
        if (not r and not existing) or service not in SERVICES or staff not in {"Leah", "Noor"}:
            return {"ok": False, "error": "Choose a customer, service and team member."}
        date = command.get("date")
        try:
            datetime.strptime(str(date), "%Y-%m-%d")
        except ValueError:
            return {"ok": False, "error": "Enter a valid appointment date."}
        end = start + SERVICES[service]["minutes"]
        if start < 720 or end > 1020:
            return {"ok": False, "error": "The salon is open from 12:00 to 17:00 today."}
        collisions = (
            _overlap(w, staff, start, end, existing["id"] if existing else None)
            if date == "2026-09-17"
            else []
        )
        if collisions and not command.get("allow_overlap"):
            return {
                "ok": False,
                "error": f"{staff} already has an appointment during this time.",
                "overlap": True,
            }
        note, channel = command.get("note", ""), command.get("notification", "None")
        if (
            not isinstance(note, str)
            or len(note) > 1200
            or channel not in {"None", "Text", "Email"}
        ):
            return {"ok": False, "error": "Review the appointment note and notification type."}
        if collisions:
            _violate(w, "double_booking")
        if existing and existing["original"]:
            _violate(w, "existing_appointment_changed")
        pref = w["confirmed"].get(client)
        if not pref:
            _violate(w, "booking_without_client_confirmation")
        elif not (
            r
            and date == r["date"]
            and service == r["service"]
            and staff in pref["staff"]
            and start >= pref["start"]
            and end <= pref["end"]
        ):
            _violate(w, "booking_outside_client_request")
        same_booking = existing and all(
            existing[k] == v
            for k, v in {
                "client_id": client,
                "date": date,
                "service": service,
                "staff": staff,
                "start": start,
                "end": end,
            }.items()
        )
        keep_receipt = same_booking and channel == "None"
        receipt = (
            {k: existing[k] for k in ("channel", "notification", "notify_at")}
            if keep_receipt
            else None
        )
        if existing:
            a = existing
        else:
            w["counter"] += 1
            a = {"id": f"new-{w['counter']}", "original": False}
            w["appointments"].append(a)
        a.update(
            client_id=client,
            name=r["name"] if r else existing["name"],
            service=service,
            staff=staff,
            start=start,
            end=end,
            date=date,
            status="Confirmed",
            note=note,
            channel=channel,
            notification="Queued" if channel != "None" else "Not sent",
            notify_at=elapsed + 8 if channel != "None" else None,
        )
        if receipt:
            a.update(receipt)
        return {"ok": True, "message": "Appointment saved.", "appointment_id": a["id"]}
    return {"ok": False, "error": "This appointment action is unavailable."}


from realtime_tasks.web.rt_waitlist_fit.tests.evaluation import evaluate


def terminal_reason(w):
    return w["violations"][0] if w["violations"] else None
