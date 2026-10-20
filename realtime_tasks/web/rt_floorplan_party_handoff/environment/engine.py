"""A researched restaurant Service surface: pre-assignments are not seated parties."""

from copy import deepcopy

import hashlib


import random

TASK_ID = "rt_floorplan_party_handoff"

CLOCK_FIELDS = ("clock_seconds", "now")

STALE_MESSAGE = "The floor has updated. Review the current assignment and try again."


def create_world(seed):
    rng = random.Random(int(hashlib.sha256(str(seed).encode()).hexdigest()[:16], 16))
    numbers = rng.sample(range(11, 59), 9)
    specs = [
        ("Dining room", "Standard", 4, 120, 135),
        ("Dining room", "Standard", 2, 260, 135),
        ("Dining room", "Standard", 4, 565, 135),
        ("Dining room", "Standard", 2, 120, 365),
        ("Dining room", "Standard", 6, 540, 365),
        ("Garden", "Outdoor", 4, 120, 135),
        ("Garden", "Outdoor", 2, 260, 135),
        ("Garden", "Outdoor", 2, 120, 365),
        ("Bar", "Bar", 6, 350, 190),
    ]
    tables = [
        dict(
            id=f"t{n}",
            number=str(n),
            room=room,
            option=option,
            capacity=cap,
            x=x,
            y=y,
            blocked=False,
            block_note="",
        )
        for n, (room, option, cap, x, y) in zip(numbers, specs)
    ]
    key = [t["id"] for t in tables]

    def party(
        pid,
        name,
        size,
        status,
        kind,
        option,
        start,
        duration,
        assigned,
        note,
        joined_at=None,
        quote=None,
    ):
        return dict(
            id=pid,
            name=name,
            size=size,
            status=status,
            kind=kind,
            option=option,
            start=start,
            duration=duration,
            tables=assigned,
            note=note,
            joined_at=joined_at,
            quote=quote,
            seated_at=start if status == "Seated" else None,
            revision=1,
            history=[],
        )

    deadline = rng.randint(86, 100)
    late_deadline = rng.randint(242, 256)
    parties = [
        party(
            "casey",
            "Casey Morgan",
            4,
            "Arrived",
            "Walk-in",
            "Standard",
            0,
            5400,
            [],
            "All four are here. An indoor standard table, please.",
            deadline - 300,
            300,
        ),
        party(
            "ellis",
            "Ellis Chen",
            6,
            "Partially arrived",
            "Walk-in",
            "Standard",
            0,
            5400,
            [key[4]],
            "Four of six are here; the other two are on their way. Indoor standard seating.",
            late_deadline - 300,
            300,
        ),
        party(
            "nina",
            "Nina Patel",
            2,
            "Expected",
            "Reservation",
            "Standard",
            1800,
            5400,
            [key[0]],
            "Standard seating booked. Arriving at 18:30.",
        ),
        party(
            "brooks",
            "Jordan Brooks",
            2,
            "Expected",
            "Reservation",
            "Outdoor",
            1800,
            5400,
            [key[5]],
            "Outdoor seating booked. Arriving at 18:30.",
        ),
        party(
            "rivera",
            "Sam Rivera",
            2,
            "Seated",
            "Reservation",
            "Standard",
            -1800,
            10800,
            [key[3]],
            "Dinner in progress.",
        ),
        party(
            "drew",
            "Alex Drew",
            2,
            "Seated",
            "Reservation",
            "Outdoor",
            -1800,
            10800,
            [key[7]],
            "Dinner in progress.",
        ),
    ]
    branch = rng.choice(["Standard", "Outdoor"])
    return dict(
        now=0.0,
        tables=tables,
        parties=parties,
        joins=[[key[0], key[1]], [key[5], key[6]]],
        room_order=["Dining room", "Garden", "Bar"],
        changes=[],
        violations=[],
        branch=branch,
        finish_at=rng.randint(100, 108),
        block_at=rng.randint(111, 117),
        arrival_at=rng.randint(128, 138),
        released="rivera" if branch == "Standard" else "drew",
        blocked_table=key[4],
        fired=[],
        seated_events=[],
    )


def _party(w, pid):
    return next((p for p in w["parties"] if p["id"] == pid), None)


def _tables(w, ids):
    return [t for t in w["tables"] if t["id"] in ids]


def _live(p):
    return p["status"] not in {"Finished", "Canceled", "Left waitlist"}


def _interval(p, now):
    start = (
        p["seated_at"]
        if p["status"] == "Seated"
        else (p["start"] if p["kind"] == "Reservation" else now)
    )
    return start, start + p["duration"]


def _conflicts(w, p, ids, now):
    start, end = _interval(p, now)
    conflicts = []
    for other in w["parties"]:
        if other["id"] == p["id"] or not _live(other) or not set(ids).intersection(other["tables"]):
            continue
        ostart, oend = _interval(other, now)
        if start < oend and ostart < end:
            conflicts.append(other["name"])
    return conflicts


def _warnings(w, p, ids, now):
    tables = _tables(w, ids)
    result = []
    if sum(t["capacity"] for t in tables) < p["size"]:
        result.append("The selected tables have fewer seats than this party.")
    if any(t["blocked"] for t in tables):
        result.append("This assignment includes a blocked table.")
    if any(t["option"] != p["option"] for t in tables):
        result.append("This assignment uses a different seating option.")
    conflicts = _conflicts(w, p, ids, now)
    if conflicts:
        result.append("This time overlaps with: " + ", ".join(conflicts) + ".")
    if len(ids) > 1 and not any(set(ids) == set(pair) for pair in w["joins"]):
        result.append(
            "These tables cannot be joined; only adjacent table pairs in the same room combine."
        )
    return result


def _safe(w, p, ids, now):
    if not ids or _warnings(w, p, ids, now):
        return False
    if len(ids) > 1 and not any(set(ids) == set(pair) for pair in w["joins"]):
        return False
    return True


def advance(w, elapsed):
    if elapsed < w["now"]:
        return
    w["now"] = float(elapsed)
    events = [(w["finish_at"], "finish"), (w["block_at"], "block"), (w["arrival_at"], "arrival")]
    for at, kind in sorted(events):
        if elapsed < at or kind in w["fired"]:
            continue
        w["fired"].append(kind)
        if kind == "finish":
            p = _party(w, w["released"])
            p["status"] = "Finished"
            p["revision"] += 1
            p["history"].append(dict(at=at, text="Finished — check closed by POS."))
            w["changes"].append(dict(at=at, text=p["name"] + " · Finished (POS)"))
        elif kind == "block":
            table = _tables(w, [w["blocked_table"]])[0]
            table.update(
                blocked=True, block_note="Ceiling leak — closed by duty manager for the evening."
            )
            w["changes"].append(
                dict(at=at, text="Table " + table["number"] + " blocked · ceiling leak")
            )
        else:
            p = _party(w, "ellis")
            p["status"] = "Arrived" if p["status"] == "Partially arrived" else p["status"]
            p["option"] = w["branch"]
            p["note"] = (
                "All six are now here. We would like "
                + ("an indoor standard table" if w["branch"] == "Standard" else "outdoor seating")
                + "; please keep us together. Updated by front host."
            )
            p["revision"] += 1
            p["history"].append(dict(at=at, text="Arrived · visit note updated by front host."))
            w["changes"].append(dict(at=at, text="Ellis Chen · Arrived · visit note updated"))


def public_world(w):
    return deepcopy(
        dict(
            restaurant="Alder House",
            date="Friday, September 18, 2026",
            clock_seconds=18 * 3600 + w["now"],
            now=w["now"],
            tables=w["tables"],
            parties=w["parties"],
            rooms=w["room_order"],
            changes=w["changes"][-8:],
        )
    )


def apply_command(w, command, now):
    kind = command.get("type")
    p = _party(w, command.get("party"))
    if p is None or not _live(p):
        return dict(ok=False, error="This guest record is no longer active.")
    if command.get("revision") != p["revision"]:
        return dict(
            ok=False,
            error="This guest record was updated. Close the assignment and review the current details.",
        )
    if kind == "assign":
        ids = command.get("tables")
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            return dict(ok=False, error="Select at least one table.")
        if len(ids) != len(set(ids)) or len(_tables(w, ids)) != len(ids):
            return dict(ok=False, error="The table selection is unavailable.")
        warnings = _warnings(w, p, ids, now)
        if warnings and command.get("override") is not True:
            return dict(ok=False, warning=True, warnings=warnings, error="Review table assignment")
        if p["status"] == "Seated" and set(ids) != set(p["tables"]):
            w["violations"].append("Moved a seated party.")
        p["tables"] = list(ids)
        p["revision"] += 1
        p["history"].append(
            dict(at=now, text="Tables assigned: " + ", ".join(t["number"] for t in _tables(w, ids)))
        )
        return dict(ok=True, message="Table assignment updated.")
    if kind == "seat":
        if p["status"] == "Seated":
            return dict(ok=True, message="This party is already seated.")
        if not p["tables"]:
            return dict(ok=False, error="Assign a table before seating this party.")
        warnings = _warnings(w, p, p["tables"], now)
        if warnings and command.get("override") is not True:
            return dict(ok=False, warning=True, warnings=warnings, error="Review seating")
        if p["status"] != "Arrived":
            w["violations"].append("Seated a party before full arrival.")
        if not _safe(w, p, p["tables"], now):
            w["violations"].append(
                "Seating violated capacity, seating request, adjacency or a reservation interval."
            )
        if any(
            o["id"] != p["id"]
            and o["status"] == "Seated"
            and set(o["tables"]).intersection(p["tables"])
            for o in w["parties"]
        ):
            w["violations"].append("Seated at an occupied table.")
        if p["kind"] == "Walk-in" and now > p["joined_at"] + p["quote"]:
            w["violations"].append("Exceeded quoted wait.")
        p.update(status="Seated", seated_at=now, revision=p["revision"] + 1)
        p["history"].append(dict(at=now, text="Seated."))
        w["seated_events"].append(dict(party=p["id"], at=now, tables=list(p["tables"])))
        return dict(ok=True, message=p["name"] + " seated.")
    if kind == "finish":
        if p["status"] != "Seated":
            return dict(ok=False, error="This party is not seated.")
        w["violations"].append("Manually finished an existing diner.")
        p.update(status="Finished", revision=p["revision"] + 1)
        return dict(ok=True, message="Party marked finished.")
    if kind == "cancel":
        w["violations"].append("Canceled a guest who must be preserved.")
        p.update(status="Canceled", revision=p["revision"] + 1)
        return dict(ok=True, message="Guest canceled.")
    return dict(ok=False, error="That action is unavailable.")


from realtime_tasks.web.rt_floorplan_party_handoff.tests.evaluation import evaluate


def terminal_reason(w):
    if w["violations"]:
        return "The seating arrangement did not meet the service request."
    for pid in ("casey", "ellis"):
        p = _party(w, pid)
        if p["status"] != "Seated" and w["now"] > p["joined_at"] + p["quote"]:
            return "A waiting party left after its quoted wait."
    return ""
