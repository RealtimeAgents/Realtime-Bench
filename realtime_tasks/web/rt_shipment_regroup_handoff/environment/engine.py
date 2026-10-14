"""Bespoke order-parent / shipment-contribution / carrier-label reconciliation.

Timeline (world seconds from 14:10:30): the mixed Ground label must be voided before packing starts
at 90 s; Sales' final paid order lands at 220 s requesting Next Day Air and the store connection
attaches it to Mara's parent order group 9031 (Ground) as its own shipment, so it has to be moved
into the Next Day Air group and both parcels labelled before the carrier handoff at 270 s.
"""

from copy import deepcopy

from datetime import datetime, timedelta

import hashlib

import math

import random

TASK_ID = "rt_shipment_regroup_handoff"

CLOCK_FIELDS = ("clock", "clock_seconds")

SCENARIOS = {TASK_ID}

STALE_MESSAGE = "Shipping records changed. Refresh and try again."

SERVICES = ("UPS Ground", "UPS Next Day Air", "UPS 2nd Day Air")

START = datetime(2026, 9, 18, 14, 10, 30)

PACKING_AT = 90

HANDOFF_AT = 270

ADDON_AT = 220

ADDON_SERVICE = SERVICES[1]

ADDON_GROUP = "9031"

ADDRESS = {
    "name": "Mara Lin",
    "company": "Alder Studio",
    "street": "18 Willow Avenue",
    "city": "Portland",
    "state": "OR",
    "zip": "97205",
    "country": "US",
}


def stamp(t):
    return (START + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")


def _order(identity, quantity, requested, created, customer="Mara Lin"):
    address = deepcopy(ADDRESS)
    if customer != "Mara Lin":
        address.update(name=customer, company="Hale Books", street="62 Cedar Street", zip="97214")
    return {
        "id": identity,
        "customer": customer,
        "address": address,
        "quantity": quantity,
        "sku": "PIN-FERN",
        "name": "Fern enamel pin",
        "price": 12,
        "requested": requested,
        "paid": quantity * 12 + 8,
        "created": stamp(created),
        "store": "MeadowPost",
    }


def _shipment(identity, owner, items, address):
    return {
        "id": identity,
        "owner": owner,
        "items": deepcopy(items),
        "address": deepcopy(address),
        "service": "UPS Ground",
        "package": "Package",
        "weight": 1.0,
        "dimensions": [10, 8, 4],
        "label": None,
        "removed": False,
        "activity": [],
    }


def _get(records, identity):
    if not isinstance(identity, str):
        return None
    return next((r for r in records if r["id"] == identity), None)


def _live(w):
    return [s for s in w["shipments"] if not s["removed"]]


def _log(s, now, message, who="Riley Chen"):
    s["activity"].append({"at": stamp(now), "by": who, "message": message})


def create_world(seed):
    rng = random.Random(int(hashlib.sha256(str(seed).encode()).hexdigest()[:16], 16))
    orders = [
        _order("9031", 2, SERVICES[0], -840),
        _order("9032", 3, SERVICES[1], -360),
        _order("8992", 1, SERVICES[0], -1500, "Owen Hale"),
    ]
    first = _shipment(
        "S201",
        "9031",
        [{"order": "9031", "quantity": 2}, {"order": "9032", "quantity": 3}],
        ADDRESS,
    )
    first["label"] = "L710001"
    _log(first, -210, "Order 9032 combined with order 9031.", "Alex Ward")
    _log(first, -120, "Label created · 1ZPW710001 · UPS Ground.", "Alex Ward")
    other = _shipment("S202", "8992", [{"order": "8992", "quantity": 1}], orders[2]["address"])
    labels = [
        {
            "id": "L710001",
            "shipment": "S201",
            "tracking": "1ZPW710001",
            "status": "Active",
            "created": stamp(-120),
            "created_s": -120,
            "voided": None,
            "items": deepcopy(first["items"]),
            "address": deepcopy(ADDRESS),
            "service": SERVICES[0],
            "price": 8.4,
        }
    ]
    rng.shuffle(orders)
    return {
        "now": 0.0,
        "orders": orders,
        "shipments": [first, other],
        "labels": labels,
        "next_shipment": 203,
        "next_label": 710002,
        "addon_id": str(rng.randrange(9040, 9090)),
        "addon_service": ADDON_SERVICE,
        "addon_quantity": rng.choice([1, 2]),
        "addon_at": ADDON_AT,
        "addon_received": False,
        "packing_checked": False,
        "old_label_used": False,
        "original_orders": deepcopy(orders),
        "unrelated_shipment": deepcopy(other),
    }


def advance(w, elapsed):
    elapsed = max(w["now"], float(elapsed))
    if not w["packing_checked"] and elapsed >= PACKING_AT:
        w["packing_checked"] = True
        w["old_label_used"] = _get(w["labels"], "L710001")["status"] != "Voided"
    if not w["addon_received"] and elapsed >= w["addon_at"]:
        o = _order(w["addon_id"], w["addon_quantity"], w["addon_service"], w["addon_at"])
        w["orders"].append(o)
        s = _shipment(
            f"S{w['next_shipment']}",
            ADDON_GROUP,
            [{"order": o["id"], "quantity": o["quantity"]}],
            o["address"],
        )
        w["next_shipment"] += 1
        _log(
            s,
            w["addon_at"],
            f"Paid order {o['id']} imported from MeadowPost and attached to "
            f"order group {ADDON_GROUP} (same recipient).",
            "System",
        )
        w["shipments"].append(s)
        w["addon_received"] = True
    w["now"] = elapsed


def public_world(w):
    shipments = deepcopy(_live(w))
    for s in shipments:
        s["status"] = "Shipped" if s["label"] else "Awaiting Shipment"
        s["requested"] = _get(w["orders"], s["owner"])["requested"]
        origins = {i["order"] for i in s["items"]}
        s["removable"] = not s["label"] and (
            not origins or (len(origins) == 1 and s["owner"] not in origins)
        )
    return {
        "clock": stamp(w["now"]),
        "clock_seconds": int(w["now"]),
        "deadlines": {"packing_starts": PACKING_AT, "handoff": HANDOFF_AT},
        "orders": deepcopy(w["orders"]),
        "shipments": shipments,
        "labels": deepcopy(w["labels"]),
        "services": list(SERVICES),
        "account": "Riley Chen",
        "store": "MeadowPost",
    }


def _bad(message):
    return {"ok": False, "error": message}


def apply_command(w, command, now):
    if not isinstance(command, dict) or not isinstance(command.get("action"), str):
        return _bad("Invalid shipping request.")
    if w["old_label_used"] or now >= HANDOFF_AT:
        return _bad("This operation is no longer available.")
    action = command["action"]
    if action == "split":
        owner = command.get("owner")
        current = [s for s in _live(w) if s["owner"] == owner and not s["label"]]
        rows = command.get("allocation")
        if not current or not isinstance(rows, list) or not rows or len(rows) > 40:
            return _bad("Select items in Awaiting Shipment.")
        expected = {}
        for s in current:
            for item in s["items"]:
                expected[item["order"]] = expected.get(item["order"], 0) + item["quantity"]
        actual, destinations, plan = {}, set(), []
        allowed = {s["id"] for s in current}
        for row in rows:
            if not isinstance(row, dict):
                return _bad("Invalid split quantities.")
            identity, qty, target = row.get("order"), row.get("quantity"), row.get("shipment")
            if (
                not isinstance(identity, str)
                or identity not in expected
                or type(qty) is not int
                or qty < 1
                or not isinstance(target, str)
                or (target not in allowed and target not in {f"new:{i}" for i in range(1, 9)})
            ):
                return _bad("Invalid split quantities or shipment.")
            actual[identity] = actual.get(identity, 0) + qty
            destinations.add(target)
            plan.append((identity, qty, target))
        if actual != expected:
            return _bad("Split quantities must equal the existing item quantities.")
        mapping = {}
        source = current[0]
        for target in sorted(destinations):
            if target.startswith("new:"):
                identity = f"S{w['next_shipment']}"
                w["next_shipment"] += 1
                s = _shipment(identity, owner, [], source["address"])
                for field in ("service", "package", "weight", "dimensions"):
                    s[field] = deepcopy(source[field])
                w["shipments"].append(s)
                mapping[target] = identity
            else:
                mapping[target] = target
        for s in current:
            s["items"] = []
        for identity, qty, target in plan:
            s = _get(w["shipments"], mapping[target])
            found = next((i for i in s["items"] if i["order"] == identity), None)
            if found:
                found["quantity"] += qty
            else:
                s["items"].append({"order": identity, "quantity": qty})
        for identity in set(mapping.values()) | allowed:
            _log(_get(w["shipments"], identity), now, "Shipment items updated using Split Ship.")
        return {"ok": True, "message": "Shipment split saved.", "created": mapping}
    if action == "combine":
        identities, survivor = command.get("shipments"), command.get("survivor")
        if (
            not isinstance(identities, list)
            or len(identities) < 2
            or len(identities) > 12
            or any(not isinstance(i, str) for i in identities)
            or len(set(identities)) != len(identities)
            or not isinstance(survivor, str)
            or survivor not in identities
        ):
            return _bad("Select at least two shipments.")
        selected = [_get(_live(w), i) for i in identities]
        if any(not s or s["label"] for s in selected):
            return _bad("Only Awaiting Shipment records can be combined.")
        base = _get(selected, survivor)
        for s in selected:
            if s is base:
                continue
            for item in s["items"]:
                found = next((i for i in base["items"] if i["order"] == item["order"]), None)
                if found:
                    found["quantity"] += item["quantity"]
                else:
                    base["items"].append(deepcopy(item))
            s["items"] = []
            s["removed"] = True
        _log(base, now, "Shipments combined: " + ", ".join(identities) + ".")
        return {"ok": True, "message": "Shipments combined.", "shipment": survivor}
    s = _get(_live(w), command.get("shipment"))
    if not s:
        return _bad("Shipment not found.")
    if action == "void_label":
        if not s["label"]:
            return _bad("This shipment has no active label.")
        label = _get(w["labels"], s["label"])
        label.update(status="Voided", voided=stamp(now))
        s["label"] = None
        _log(
            s,
            now,
            "Label " + label["tracking"] + " voided. Shipment returned to Awaiting Shipment.",
        )
        return {"ok": True, "message": "Label voided. Refunds follow the carrier's policy."}
    if s["label"]:
        return _bad("This shipment has already been labelled.")
    if action == "remove_shipment":
        origins = {i["order"] for i in s["items"]}
        if not origins:
            s["removed"] = True
            return {"ok": True, "message": "Empty shipment removed.", "owner": s["owner"]}
        if len(origins) == 1 and s["owner"] not in origins:
            old = s["owner"]
            s["owner"] = next(iter(origins))
            _log(s, now, f"Uncombined from order {old}; shipment returned to order {s['owner']}.")
            return {
                "ok": True,
                "owner": s["owner"],
                "message": f"Order {s['owner']} uncombined from order {old}.",
            }
        return _bad(
            "Only a shipment holding another order's items can be uncombined from this order."
        )
    if action == "configure":
        service, package = command.get("service"), command.get("package", s["package"])
        weight = command.get("weight", s["weight"])
        dims = command.get("dimensions", s["dimensions"])
        if (
            service not in SERVICES
            or package not in ("Package", "Custom box")
            or type(weight) not in (int, float)
            or not math.isfinite(weight)
            or not 0 < weight <= 100
            or not isinstance(dims, list)
            or len(dims) != 3
            or any(
                type(d) not in (int, float) or not math.isfinite(d) or not 0 < d <= 60 for d in dims
            )
        ):
            return _bad("Enter valid service, weight and package dimensions.")
        s.update(service=service, package=package, weight=weight, dimensions=deepcopy(dims))
        _log(s, now, "Shipping configuration updated: " + service + ".")
        return {"ok": True, "message": "Shipping configuration saved."}
    if action == "create_label":
        identity = f"L{w['next_label']}"
        tracking = f"1ZPW{w['next_label']}"
        w["next_label"] += 1
        label = {
            "id": identity,
            "shipment": s["id"],
            "tracking": tracking,
            "status": "Active",
            "created": stamp(now),
            "created_s": now,
            "voided": None,
            "items": deepcopy(s["items"]),
            "address": deepcopy(s["address"]),
            "service": s["service"],
            "price": 8.4
            if s["service"] == SERVICES[0]
            else 24.6
            if s["service"] == SERVICES[1]
            else 17.1,
        }
        w["labels"].append(label)
        s["label"] = identity
        _log(s, now, "Label created · " + tracking + " · " + s["service"] + ".")
        return {"ok": True, "message": "Label created.", "label": identity}
    return _bad("Unknown shipping operation.")


from realtime_tasks.web.rt_shipment_regroup_handoff.tests.evaluation import evaluate


def terminal_reason(w):
    if w["old_label_used"]:
        return "Packing began with the unvoided mixed-shipment label."
    if w["now"] >= HANDOFF_AT and not evaluate(w)["resolved"]:
        return "The parcel preparation handoff has passed."
    return None
