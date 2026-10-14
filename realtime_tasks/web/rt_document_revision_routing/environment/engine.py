"""A researched document revision and causal recipient-notification workflow."""

from copy import deepcopy

from datetime import datetime

import random

import re

TASK_ID = "rt_document_revision_routing"

CLOCK_FIELDS = ("elapsed",)

STALE_MESSAGE = "The document changed. Refresh and try again."

VARIABLES = ("Course.Date", "Course.Venue", "Course.Fee", "Course.Attendees")

CLIENT_SIGNING_AT = 90

SEND_DEADLINE = 240

EPISODE_END = 300


def _activity(w, text, at):
    w["activity"].append({"text": text, "at": at})


def _enqueue(w, at, kind, **data):
    w["_counter"] += 1
    w["_events"].append(dict(at=at, kind=kind, seq=w["_counter"], **data))


def _person(w, ident):
    return next(p for p in w["people"] if p["id"] == ident)


def _revision(w):
    return next((r for r in w["_revisions"] if r["number"] == w["revision"]), None)


def _notify(w, r, ident, at):
    item = r["recipients"][ident]
    if item["notified"] is not None:
        return
    w["_counter"] += 1
    item.update(notified=at, notified_seq=w["_counter"], status="Not viewed")
    _activity(w, f"Email delivered to {_person(w, ident)['name']} · Version {r['number']}", at)
    _enqueue(w, at + 3, "view", revision=r["number"], person=ident)
    if _person(w, ident)["kind"] == "Signer":
        _enqueue(w, at + 20, "finish", revision=r["number"], person=ident)


def _distribute(w, r, at):
    if not r["sequential"]:
        for ident in r["order"]:
            _notify(w, r, ident, at)
        return
    for ident in r["order"]:
        _notify(w, r, ident, at)
        if _person(w, ident)["kind"] == "Signer" and r["recipients"][ident]["signed"] is None:
            break


def create_world(seed):
    rng = random.Random(str(seed))
    count = rng.choice([14, 18, 22])
    venues = ["Birch Studio", "Quay Gallery", "Garden Workshop", "Maple Hall"]
    rng.shuffle(venues)
    shift = rng.choice([0, 1, 2])
    dates = [f"{d + shift} October 2026" for d in (12, 13, 15, 16)]
    fees = [
        rng.choice([880, 920, 960]),
        rng.choice([1380, 1420, 1460]),
        rng.choice([1100, 1140, 1180]),
        rng.choice([1200, 1220, 1240]),
    ]
    offers = [
        dict(venue=v, date=d, seats=n, fee=f)
        for v, d, n, f in zip(venues, dates, [16, 24, 20, 26], fees)
    ]
    wanted = next(o for o in offers if o["seats"] >= count and o["fee"] <= 1250)
    rng.shuffle(offers)
    order = ["lead", "client", "accounts"]
    rng.shuffle(order)
    schedule = ["roster", "venues"]
    rng.shuffle(schedule)
    w = dict(
        elapsed=0,
        title="Northline Tools · Team training",
        owner="Morgan Hall",
        workspace="Ridgeway Skills",
        status="Viewed",
        revision=1,
        variables={
            "Course.Date": "9 October 2026",
            "Course.Venue": "Cedar Room",
            "Course.Fee": "800",
            "Course.Attendees": "10",
        },
        people=[
            dict(
                id="lead",
                name="Ada Brooks",
                email="ada@ridgewayskills.example",
                role="Delivery lead",
                kind="Signer",
            ),
            dict(
                id="client",
                name="Casey Wu",
                email="casey@northlinetools.example",
                role="Client",
                kind="Signer",
            ),
            dict(
                id="accounts",
                name="Jamie Cole",
                email="accounts@northlinetools.example",
                role="Accounts",
                kind="CC",
            ),
        ],
        order=order,
        sequential=False,
        comments=[
            dict(
                id="c1",
                author="Sam Patel",
                at=-60,
                private=True,
                text="The client is updating the attendance list. I am confirming their roster; Laila is checking room dates and fees.",
                resolved=False,
            )
        ],
        activity=[
            dict(at=-150, text="Morgan Hall created Version 1"),
            dict(at=-80, text="Morgan Hall sent Version 1"),
            dict(at=-40, text="Ada Brooks signed Version 1"),
            dict(at=-10, text="Casey Wu viewed Version 1"),
        ],
        _events=[],
        _counter=0,
        _revisions=[],
        _old_completed=False,
        _withdrawn_at=None,
        _offers=offers,
        _count=count,
        _target={
            "Course.Date": wanted["date"],
            "Course.Venue": wanted["venue"],
            "Course.Fee": str(wanted["fee"]),
            "Course.Attendees": str(count),
        },
    )
    r = dict(
        number=1,
        active=True,
        sent=-80,
        variables=deepcopy(w["variables"]),
        sequential=False,
        order=list(order),
        recipients={
            p["id"]: dict(
                notified=-80,
                notified_seq=-4,
                signed=-40 if p["id"] == "lead" else None,
                signed_seq=-2 if p["id"] == "lead" else None,
                status="Signed"
                if p["id"] == "lead"
                else "Viewed"
                if p["id"] == "client"
                else "Not viewed",
            )
            for p in w["people"]
        },
    )
    w["_revisions"].append(r)
    _enqueue(w, CLIENT_SIGNING_AT, "finish", revision=1, person="client")
    for at, kind in zip((110, 130), schedule):
        _enqueue(w, at, kind)
    return w


def advance(w, elapsed):
    elapsed = max(w["elapsed"], min(float(elapsed), EPISODE_END))
    while w["_events"]:
        event = min(w["_events"], key=lambda e: (e["at"], e["seq"]))
        if event["at"] > elapsed:
            break
        w["_events"].remove(event)
        at = event["at"]
        kind = event["kind"]
        if kind in ("roster", "venues"):
            if w["status"] == "Completed":
                continue
            if kind == "roster":
                text = f"Final roster confirmed by Casey: {w['_count']} participants for the full session. The earlier ten-person estimate is superseded."
                author = "Sam Patel"
            else:
                text = "Confirmed room offers, including training and room hire:\n" + "\n".join(
                    f"{o['date']} — {o['venue']} — {o['seats']} seats — ${o['fee']} total"
                    for o in w["_offers"]
                )
                author = "Laila Ross"
            w["comments"].append(
                dict(
                    id=f"c{len(w['comments']) + 1}",
                    author=author,
                    at=at,
                    private=True,
                    text=text,
                    resolved=False,
                )
            )
            continue
        r = next(r for r in w["_revisions"] if r["number"] == event["revision"])
        if not r["active"] or r["number"] != w["revision"]:
            continue
        item = r["recipients"][event["person"]]
        if kind == "view" and item["signed"] is None:
            item["status"] = "Viewed"
            if w["status"] == "Sent":
                w["status"] = "Viewed"
            _activity(w, f"{_person(w, event['person'])['name']} viewed Version {r['number']}", at)
        elif kind == "finish" and item["signed"] is None:
            w["_counter"] += 1
            item.update(signed=at, signed_seq=w["_counter"], status="Signed")
            _activity(w, f"{_person(w, event['person'])['name']} signed Version {r['number']}", at)
            if r["number"] == 1:
                w["_old_completed"] = True
            _distribute(w, r, at)
            if all(
                r["recipients"][p["id"]]["signed"] is not None
                for p in w["people"]
                if p["kind"] == "Signer"
            ):
                w["status"] = "Completed"
                _activity(w, f"All signers completed Version {r['number']}", at)
    w["elapsed"] = elapsed


def public_world(w):
    r = _revision(w)
    recipients = []
    for ident in w["order"]:
        p = deepcopy(_person(w, ident))
        item = r["recipients"][ident] if r and w["status"] != "Draft" else None
        p.update(
            status=item["status"] if item else "Not sent",
            sent=item["notified"] if item else None,
            signed=item["signed"] if item else None,
        )
        recipients.append(p)
    return {
        **{
            k: deepcopy(w[k])
            for k in (
                "title",
                "owner",
                "workspace",
                "status",
                "revision",
                "variables",
                "order",
                "sequential",
                "comments",
                "activity",
            )
        },
        "recipients": recipients,
        "deadlines": {
            "client_signing": CLIENT_SIGNING_AT,
            "send_by": SEND_DEADLINE,
            "verify_by": EPISODE_END,
        },
        "elapsed": w["elapsed"],
    }


def apply_command(w, command, elapsed):
    op = command.get("op")
    if op == "edit":
        if w["status"] != "Draft":
            r = _revision(w)
            if r:
                r["active"] = False
            if w["_withdrawn_at"] is None:
                w["_withdrawn_at"] = elapsed
            w["revision"] += 1
            w["status"] = "Draft"
            _activity(
                w,
                f"Morgan Hall started Version {w['revision']} · Signature fields cleared",
                elapsed,
            )
        return {"ok": True, "message": "Document is now a draft."}
    if op in ("variable", "send") and w["status"] != "Draft":
        return {
            "ok": False,
            "error": "This document is not editable. Open Edit to create a new version.",
        }
    if op == "variable":
        name, value = command.get("name"), command.get("value")
        if name not in VARIABLES or not isinstance(value, str) or len(value) > 100:
            return {"ok": False, "error": "Enter a valid variable value."}
        w["variables"][name] = value.strip()
        return {"ok": True, "message": "Changes saved."}
    if op == "recipients":
        order = command.get("order")
        sequential = command.get("sequential")
        if (
            not isinstance(order, list)
            or len(order) != 3
            or any(not isinstance(x, str) for x in order)
            or set(order) != {"lead", "client", "accounts"}
            or type(sequential) is not bool
        ):
            return {"ok": False, "error": "The recipient list could not be saved."}
        if w["status"] != "Draft":
            r = _revision(w)
            if w["status"] not in ("Sent", "Viewed") or not r:
                return {"ok": False, "error": "This recipient list is no longer editable."}
            fixed = [i for i in w["order"] if r["recipients"][i]["notified"] is not None]
            if sequential != w["sequential"] or order[: len(fixed)] != fixed:
                return {
                    "ok": False,
                    "error": "Recipients who already received the document cannot be reordered.",
                }
            r["order"] = list(order)
        w["order"] = list(order)
        w["sequential"] = sequential
        return {"ok": True, "message": "Recipient changes saved."}
    if op == "send":
        subject = command.get("subject", "")
        message = command.get("message", "")
        if (
            not isinstance(subject, str)
            or not subject.strip()
            or len(subject) > 250
            or not isinstance(message, str)
            or len(message) > 10000
        ):
            return {"ok": False, "error": "Enter an email subject and a valid message."}
        if any(not w["variables"][k] for k in VARIABLES):
            return {"ok": False, "error": "Fill in all document variables before sending."}
        r = dict(
            number=w["revision"],
            active=True,
            sent=elapsed,
            variables=deepcopy(w["variables"]),
            sequential=w["sequential"],
            order=list(w["order"]),
            subject=subject,
            message=message,
            recipients={
                p["id"]: dict(
                    notified=None, notified_seq=None, signed=None, signed_seq=None, status="Waiting"
                )
                for p in w["people"]
            },
        )
        w["_revisions"].append(r)
        w["status"] = "Sent"
        _activity(w, f"Morgan Hall sent Version {w['revision']}", elapsed)
        _distribute(w, r, elapsed)
        return {"ok": True, "message": "Document sent."}
    if op == "comment":
        comment = next((c for c in w["comments"] if c["id"] == command.get("id")), None)
        if not comment or type(command.get("resolved")) is not bool:
            return {"ok": False, "error": "This comment could not be updated."}
        comment["resolved"] = command["resolved"]
        return {"ok": True, "message": "Comment updated."}
    if op == "remind":
        ident = command.get("person")
        r = _revision(w)
        if (
            ident not in ("lead", "client", "accounts")
            or not r
            or w["status"] not in ("Sent", "Viewed")
            or r["recipients"][ident]["notified"] is None
            or r["recipients"][ident]["signed"] is not None
        ):
            return {"ok": False, "error": "A reminder is not available for this recipient."}
        _activity(w, f"Morgan Hall sent a reminder to {_person(w, ident)['name']}", elapsed)
        return {"ok": True, "message": "Reminder sent."}
    return {"ok": False, "error": "This operation is unavailable."}


def _same_values(values, target):
    def normalized(k, v):
        if k == "Course.Date":
            for fmt in (
                "%d %B %Y",
                "%d %b %Y",
                "%B %d, %Y",
                "%b %d, %Y",
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%d/%m/%Y",
            ):
                try:
                    return datetime.strptime(v.strip(), fmt).date()
                except ValueError:
                    pass
            return v.strip().casefold()
        if k in ("Course.Fee", "Course.Attendees"):
            try:
                return float(v.replace("$", "").replace(",", ""))
            except (ValueError, AttributeError):
                return None
        return re.sub(r"\s+", " ", v.strip()).casefold()

    return all(normalized(k, values[k]) == normalized(k, target[k]) for k in VARIABLES)


def _safe_routing(r):
    lead, client, accounts = [r["recipients"][k] for k in ("lead", "client", "accounts")]
    if client["notified"] is not None and (
        lead["signed_seq"] is None or client["notified_seq"] <= lead["signed_seq"]
    ):
        return False
    if accounts["notified"] is not None and any(
        p["signed_seq"] is None or accounts["notified_seq"] <= p["signed_seq"]
        for p in (lead, client)
    ):
        return False
    return True


from realtime_tasks.web.rt_document_revision_routing.tests.evaluation import evaluate


def terminal_reason(w):
    return (
        "The handover time has arrived."
        if w["elapsed"] >= EPISODE_END and not evaluate(w)["resolved"]
        else None
    )
