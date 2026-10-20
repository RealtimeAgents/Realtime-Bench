"""Order holds and shipping-address corrections with scheduled warehouse collections.

A hold withdraws the fulfillment request. Releasing the order submits its current
address, so later corrections require another hold, edit, and release.
"""

import hashlib

import math

import random

import re

from copy import deepcopy

TASK_ID = "rt_dispatch_address_hold"

CLOCK_FIELDS = ("clock_seconds", "next_collection")

FIRST_COLLECTION_AT = 60.0

LAST_COLLECTION_AT = 195.0

CORRECTION_AT = 150.0

COLLECTION_TIMES = {FIRST_COLLECTION_AT: "12:01:00", LAST_COLLECTION_AT: "12:03:15"}

HOLD_REASONS = ("Address issue", "Awaiting payment", "Inventory out of stock", "Other")

SUPPORT_AUTHOR = "Lea · Support"

STREET_SUFFIXES = {"ln": "lane", "st": "street", "ave": "avenue", "rd": "road"}


def _ship_to(address):
    return f"{address['street']}, Apt {address['unit']}, {address['city']} OR {address['postal']}"


def create_world(episode_key):
    rng = random.Random(int(hashlib.sha256(episode_key.encode()).hexdigest(), 16))
    streets = rng.sample(["18 Willow Lane", "42 Alder Street", "76 Elm Avenue", "31 Birch Road"], 3)
    number = rng.randint(2, 8)
    first_letter, corrected_letter = rng.sample(["A", "B", "C"], 2)
    shipping = {"street": streets[0], "unit": "2A", "city": "Portland", "postal": "97205"}
    confirmed = {
        "street": streets[2],
        "unit": f"{number}{first_letter}",
        "city": "Portland",
        "postal": "97205",
    }
    return {
        "shipping": shipping,
        "confirmed": confirmed,
        # The fixed branch: only the apartment letter changes in Lea's correction.
        "corrected": {**confirmed, "unit": f"{number}{corrected_letter}"},
        "tentative": streets[1],
        "confirmation_at": float(rng.randint(88, 104)),
        "tentative_at": float(rng.randint(32, 40)),
        "correction_at": CORRECTION_AT,
        "pickup_at": LAST_COLLECTION_AT,
        "elapsed": 0.0,
        "held": False,
        "hold_reason": "",
        "hold_at": None,
        "release_at": None,
        # The fulfillment request the warehouse labels from; None while the order is on hold.
        "request": {"at": 0.0, "shipping": deepcopy(shipping)},
        "stage": 0,
        "dispatch": None,
        "blocked_pickup": False,
        "violations": [],
        "history": [
            {
                "at": 0,
                "author": SUPPORT_AUTHOR,
                "text": "Mara is checking the delivery address. I am speaking with her now.",
            },
            {
                "at": 0,
                "author": "Warehouse",
                "text": f"Fulfillment request received for {_ship_to(shipping)}. We pick and label from the request on file: "
                "a hold withdraws it, releasing the hold sends a new request with the order's current shipping address.",
            },
            {
                "at": 0,
                "author": "Warehouse",
                "text": "Collections today at 12:01:00 and 12:03:15 (last). Orders that are not on hold are handed to the carrier at each collection.",
            },
            {
                "at": 0,
                "author": "Juniper Supply",
                "text": "Payment of $68.00 received. 2 linen cushion covers awaiting fulfillment.",
            },
        ],
        "commands": [],
        "message": "",
        "expired": False,
    }


def _note(world, at, author, text):
    world["history"].insert(0, {"at": round(at, 2), "author": author, "text": text})


def advance(world, elapsed):
    world["elapsed"] = elapsed
    if world["dispatch"] or world["expired"]:
        return
    if world["stage"] < 1 and elapsed >= world["tentative_at"]:
        world["stage"] = 1
        _note(
            world,
            world["tentative_at"],
            SUPPORT_AUTHOR,
            f"Mara mentioned {world['tentative']}. She has not confirmed the apartment or whether to use this address yet.",
        )
    if not world["blocked_pickup"] and elapsed >= FIRST_COLLECTION_AT:
        world["blocked_pickup"] = True
        if not world["held"]:
            _dispatch(world, FIRST_COLLECTION_AT)
            return
        _note(
            world,
            FIRST_COLLECTION_AT,
            "Warehouse",
            "Collection completed. This order remains on hold.",
        )
    if world["stage"] < 2 and elapsed >= world["confirmation_at"]:
        world["stage"] = 2
        address = world["confirmed"]
        _note(
            world,
            world["confirmation_at"],
            SUPPORT_AUTHOR,
            f"Mara has now confirmed: {address['street']}, Apt {address['unit']}, Portland OR 97205. The earlier address was for her office. She wants this order delivered to her apartment.",
        )
    if world["stage"] < 3 and elapsed >= world["correction_at"]:
        world["stage"] = 3
        _note(
            world,
            world["correction_at"],
            SUPPORT_AUTHOR,
            f"Correction from Mara: the apartment is {world['corrected']['unit']}, not {world['confirmed']['unit']}. "
            f"She read {world['confirmed']['unit']} off her old lease; the street and ZIP are unchanged. "
            f"Please make sure the parcel is not addressed to {world['confirmed']['unit']}.",
        )
    if elapsed >= world["pickup_at"]:
        if world["held"]:
            world["expired"] = True
            _note(
                world,
                world["pickup_at"],
                "Warehouse",
                "Today's collections have finished. The order is still on hold.",
            )
        else:
            _dispatch(world, world["pickup_at"])


def _dispatch(world, at):
    # The carrier label comes from the request on file, not from the address as it reads now.
    shipping = deepcopy(world["request"]["shipping"])
    world["dispatch"] = {"at": at, "shipping": shipping, "tracking": "JN1048PK"}
    world["message"] = "Parcel handed to carrier."
    _note(
        world,
        at,
        "Warehouse",
        f"2 items fulfilled. Parcel handed to carrier for {_ship_to(shipping)}. Tracking JN1048PK.",
    )


def _next_collection(world):
    """The upcoming collection as the header shows it; None once today's collections are over."""
    for at in (FIRST_COLLECTION_AT, world["pickup_at"]):
        if world["elapsed"] < at:
            return {
                "time": COLLECTION_TIMES[at],
                "in_seconds": math.ceil(at - world["elapsed"]),
                "last": at == world["pickup_at"],
            }
    return None


def _new_comment(world):
    """Lea's newest comment since the operator's last address edit; None when there is none.

    The page announces it above the shipment card until the address is edited (or the operator
    opens it from the strip, which the browser remembers on its own). It is derived from history
    and commands only, both already versioned, so it never changes with the clock and is not a
    CLOCK_FIELD.
    """
    edited_at = max(
        (c["at"] for c in world["commands"] if c["operation"] == "save_shipping"), default=0.0
    )
    return deepcopy(
        next(
            (
                item
                for item in world["history"]  # newest first
                if item["author"] == SUPPORT_AUTHOR and item["at"] > edited_at
            ),
            None,
        )
    )


def public_world(world):
    request = world["request"]
    return {
        "shipping": deepcopy(world["shipping"]),
        "held": world["held"],
        "hold_reason": world["hold_reason"],
        "fulfillment": "Fulfilled"
        if world["dispatch"]
        else "On hold"
        if world["held"]
        else "Unfulfilled",
        "request": {"at": int(request["at"]), "shipping": deepcopy(request["shipping"])}
        if request
        else None,
        "request_stale": bool(
            request and _address_key(request["shipping"]) != _address_key(world["shipping"])
        ),
        "last_collection": COLLECTION_TIMES[world["pickup_at"]],
        "tracking": world["dispatch"]["tracking"] if world["dispatch"] else None,
        "history": deepcopy(world["history"]),
        "message": world["message"],
        "new_comment": _new_comment(world),
        "clock_seconds": int(world["elapsed"]),
        "next_collection": _next_collection(world),
    }


def apply_command(world, command, elapsed):
    operation = command.get("operation")
    if operation == "hold":
        reason = command.get("reason")
        if world["held"] or command.get("items") != ["linen-cover"]:
            return {"ok": False, "error": "Select the unfulfilled items to hold."}
        if reason not in HOLD_REASONS:
            return {"ok": False, "error": "Select a hold reason."}
        world["held"] = True
        world["hold_reason"] = reason
        world["hold_at"] = elapsed if world["hold_at"] is None else world["hold_at"]
        world["request"] = None
        _note(world, elapsed, "You", f"Fulfillment placed on hold. {reason}.")
        _note(
            world,
            elapsed,
            "Warehouse",
            "Fulfillment request withdrawn. Nothing is picked for this order while it is on hold.",
        )
        message = "Fulfillment is on hold. The warehouse request has been withdrawn."
    elif operation == "save_shipping":
        shipping = command.get("shipping")
        if not isinstance(shipping, dict) or set(shipping) != {"street", "unit", "city", "postal"}:
            return {"ok": False, "error": "Check the address fields."}
        if any(not isinstance(v, str) or not v.strip() or len(v) > 120 for v in shipping.values()):
            return {"ok": False, "error": "Complete all address fields."}
        world["shipping"] = {k: v.strip() for k, v in shipping.items()}
        # The field already means apartment; storing a repeated label must not alter the unit.
        world["shipping"]["unit"] = (
            re.sub(
                r"^(?:apartment\b|apt\.?\s*|unit\b|#)\s*",
                "",
                world["shipping"]["unit"],
                flags=re.IGNORECASE,
            )
            .strip()
            .upper()
        )
        _note(world, elapsed, "You", "Shipping address updated.")
        message = "Shipping address saved."
        if world["request"] and _address_key(world["request"]["shipping"]) != _address_key(
            world["shipping"]
        ):
            message = "Shipping address saved. The request at the warehouse still carries the previous address."
    elif operation == "release":
        if not world["held"]:
            return {"ok": False, "error": "There is no fulfillment hold to release."}
        if elapsed < world["confirmation_at"]:
            world["violations"].append("released_during_customer_check")
        world["held"] = False
        world["hold_reason"] = ""
        world["release_at"] = elapsed
        world["request"] = {"at": elapsed, "shipping": deepcopy(world["shipping"])}
        _note(world, elapsed, "You", "Fulfillment hold released.")
        _note(
            world,
            elapsed,
            "Warehouse",
            f"Fulfillment request received for {_ship_to(world['shipping'])}. The label prints from this request.",
        )
        message = f"Fulfillment released. Request sent to the warehouse for {_ship_to(world['shipping'])}."
    elif operation == "comment":
        comment = command.get("text")
        if not isinstance(comment, str) or not comment.strip() or len(comment) > 2000:
            return {"ok": False, "error": "Enter a comment of up to 2,000 characters."}
        _note(world, elapsed, "Alex · Store operations", comment.strip())
        message = "Comment added to the order timeline."
    else:
        return {"ok": False, "error": "Unknown operation."}
    world["message"] = message
    world["commands"].append({"at": elapsed, "operation": operation})
    return {"ok": True, "message": message}


def _address_key(address):
    """Case, punctuation, spacing and common street suffixes do not change where a parcel goes."""
    key = {}
    for field, value in address.items():
        words = re.sub(r"[^a-z0-9]+", " ", value.lower()).split()
        key[field] = " ".join(STREET_SUFFIXES.get(word, word) for word in words)
    return key


from realtime_tasks.web.rt_dispatch_address_hold.tests.evaluation import evaluate


def terminal_reason(world):
    if world["dispatch"]:
        evaluation = evaluate(world)
        if evaluation["resolved"]:
            return None
        if evaluation["address_matches_first_confirmation"]:
            return "order_dispatched_with_superseded_address"
        return "order_dispatched_without_required_address_check"
    if world["expired"]:
        return "collection_closed"
    return None
