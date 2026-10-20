"""Bespoke Fieldroom catalog episode."""

from copy import deepcopy

import hashlib

import itertools

import random

TASK_ID = "rt_bundle_catalog_rebind"

CLOCK_FIELDS = ("now",)

STALE_MESSAGE = "The catalog changed. Reload the current product before saving."

FIRST_ORDERS_AT = 135.0

CAMPAIGN_ENDS_AT = 285.0

RENAME_AT = 40.0

PRICE_AT = 175.0

LAST_ORDER_AT = 258.0


def create_world(episode_id):
    rng = random.Random(int(hashlib.sha256(episode_id.encode()).hexdigest()[:16], 16))
    colors = rng.sample(["Ink", "Clay", "Sage", "Ocean", "Sand", "Plum"], 3)
    products = {}
    for key, title, base in [("mat", "Desk mat", 2400), ("pouch", "Cable pouch", 1200)]:
        variants = [
            {
                "id": f"{key}-{i}",
                "color": color,
                "price": base + rng.randrange(0, 5) * 100,
                "stock": 20 + i * 3,
            }
            for i, color in enumerate(colors)
        ]
        rng.shuffle(variants)
        products[key] = {
            "id": key,
            "title": title,
            "option": "Color",
            "variants": variants,
            "revision": 1,
            "status": "active",
            "updated": "Today at 15:55",
        }
    parent = {
        "id": "desk-pair",
        "title": "Desk pair",
        "description": "A desk mat and a cable pouch, made for a tidier working day.",
        "status": "active",
        "online": True,
        "combined": False,
        "components": {
            key: {"quantity": 1, "selected": {v["id"]: v["color"] for v in p["variants"]}}
            for key, p in products.items()
        },
        "prices": {},
    }
    w = {
        "now": 0.0,
        "products": products,
        "parent": parent,
        "invalidated": [],
        "orders": [],
        "other": {"title": "Pen rest", "status": "active", "price": 900},
        "original_other": {"title": "Pen rest", "status": "active", "price": 900},
        "original_orders": [{"id": "#8100", "title": "Pen rest", "quantity": 1, "total": 900}],
        "events_done": [],
        "first_at": FIRST_ORDERS_AT,
        "rename_at": RENAME_AT,
        "price_at": PRICE_AT,
        "last_at": LAST_ORDER_AT,
        "rename_product": "pouch",
        "rename_color": rng.choice(colors),
        "new_color": rng.choice(
            [x for x in ["Ink", "Clay", "Sage", "Ocean", "Sand", "Plum"] if x not in colors]
        ),
        "price_steps": rng.sample([-300, 200, 400], 2),
        "catalog_prices": {v["id"]: v["price"] for p in products.values() for v in p["variants"]},
    }
    for row in variant_rows(w):
        parent["prices"][row["key"]] = row["sum"] - 500
    return w


def current_selected(w, key):
    product = w["products"][key]
    saved = w["parent"]["components"].get(key, {}).get("selected", {})
    return [v for v in product["variants"] if saved.get(v["id"]) == v["color"]]


def variant_rows(w):
    p = w["parent"]
    keys = sorted(p["components"])
    if not keys:
        return []
    rows = []
    for parts in itertools.product(*(current_selected(w, key) for key in keys)):
        same = len({v["color"] for v in parts}) == 1
        if p["combined"] and not same:
            continue
        identifiers = "|".join(v["id"] for v in parts)
        content = [
            {
                "product": key,
                "title": w["products"][key]["title"],
                "variant": v["id"],
                "color": v["color"],
                "quantity": p["components"][key]["quantity"],
                "unit_price": v["price"],
            }
            for key, v in zip(keys, parts)
        ]
        rows.append(
            {
                "key": identifiers,
                "title": parts[0]["color"]
                if p["combined"]
                else " / ".join(v["color"] for v in parts),
                "components": content,
                "sum": sum(x["unit_price"] * x["quantity"] for x in content),
                "price": p["prices"].get(identifiers, 0),
                "available": min(
                    v["stock"] // p["components"][key]["quantity"] for key, v in zip(keys, parts)
                ),
            }
        )
    return rows


def shared_colors(w):
    return set.intersection(
        *(set(v["color"] for v in p["variants"]) for p in w["products"].values())
    )


def correct_listing(w):
    p = w["parent"]
    rows = variant_rows(w)
    return (
        components_preserved(w)
        and not w["invalidated"]
        and set(p["components"]) == {"mat", "pouch"}
        and p["combined"]
        and all(p["components"][k]["quantity"] == 1 for k in p["components"])
        and {r["title"] for r in rows} == shared_colors(w)
        and len(rows) == len(shared_colors(w))
        and all(r["price"] == r["sum"] - 500 for r in rows)
    )


def purchasable(w):
    return (
        w["parent"]["status"] in {"active", "unlisted"}
        and w["parent"]["online"]
        and not w["invalidated"]
        and any(r["available"] > 0 for r in variant_rows(w))
    )


def purchase(w, label, at):
    rows = [r for r in variant_rows(w) if r["available"] > 0]
    entry = {
        "id": "#8101" if label == "first" else "#8102",
        "at": at,
        "customer": "Riley West" if label == "first" else "Alex Moore",
        "status": "Paid",
        "label": label,
        "listing_correct": correct_listing(w),
        "missed": not purchasable(w),
        "components": [],
        "total": 0,
        "product": "Desk pair",
    }
    if not entry["missed"]:
        # Synthetic shoppers can choose unlike colors when the listing offers them independently.
        row = next((r for r in rows if len({v["color"] for v in r["components"]}) > 1), rows[-1])
        entry.update(components=deepcopy(row["components"]), total=row["price"])
        for component in row["components"]:
            v = next(
                v
                for v in w["products"][component["product"]]["variants"]
                if v["id"] == component["variant"]
            )
            v["stock"] -= component["quantity"]
    w["orders"].append(entry)


def advance(w, elapsed):
    end = max(w["now"], min(300.0, float(elapsed)))
    events = [
        (w["rename_at"], "rename"),
        (w["first_at"], "first"),
        (w["price_at"], "price"),
        (w["last_at"], "last"),
    ]
    for at, event in sorted(events):
        if event in w["events_done"] or at > end:
            continue
        w["now"] = at
        if event == "rename":
            key = w["rename_product"]
            product = w["products"][key]
            variant = next(v for v in product["variants"] if v["color"] == w["rename_color"])
            previous = variant["color"]
            variant["color"] = w["new_color"]
            product["revision"] += 1
            product["updated"] = f"Today at 16:00:{int(at):02}"
            saved = w["parent"]["components"].get(key, {}).get("selected", {})
            if variant["id"] in saved:
                w["invalidated"] = [
                    {"product": key, "previous": previous, "variant": variant["id"]}
                ]
                w["parent"]["status"] = "draft"
        elif event == "price":
            colors = sorted(shared_colors(w))
            product = w["products"]["mat"]
            product["revision"] += 1
            for color, step in zip(colors, w["price_steps"]):
                v = next(v for v in product["variants"] if v["color"] == color)
                w["catalog_prices"][v["id"]] += step
                v["price"] = w["catalog_prices"][v["id"]]
            product["updated"] = f"Today at 16:{int(at) // 60:02}:{int(at) % 60:02}"
        else:
            purchase(w, event, at)
        w["events_done"].append(event)
    w["now"] = end


def public_world(w):
    parent = deepcopy(w["parent"])
    parent["variants"] = variant_rows(w)
    # Component retail totals are not a product-calculated recommendation or benchmark operand.
    for row in parent["variants"]:
        row.pop("sum")
        for c in row["components"]:
            c.pop("unit_price")
    parent.pop("prices")
    orders = [
        {k: deepcopy(v) for k, v in o.items() if k not in {"label", "listing_correct", "missed"}}
        for o in w["orders"]
        if not o["missed"]
    ]
    for order in orders:
        for c in order["components"]:
            c.pop("unit_price", None)
    return {
        "now": w["now"],
        "first_orders_at": FIRST_ORDERS_AT,
        "campaign_ends_at": CAMPAIGN_ENDS_AT,
        "store": "Fieldroom Goods",
        "user": "Morgan Vale",
        "parent": parent,
        "products": deepcopy(w["products"]),
        "notices": deepcopy(w["invalidated"]),
        "orders": deepcopy(w["original_orders"]) + orders,
        "other": deepcopy(w["other"]),
    }


def money(value):
    if type(value) is not int or not 0 <= value <= 1000000:
        raise ValueError
    return value


def apply_command(w, c, elapsed):
    op = c.get("op")
    p = w["parent"]
    if op == "composition":
        items = c.get("components")
        combined = c.get("combined")
        if (
            not isinstance(items, dict)
            or not items
            or set(items) - set(w["products"])
            or type(combined) is not bool
        ):
            return {"ok": False, "error": "Select at least one product."}
        prepared = {}
        value_sets = []
        for key, item in items.items():
            if not isinstance(item, dict):
                return {"ok": False, "error": "Product selection is invalid."}
            quantity = item.get("quantity")
            selected = item.get("selected")
            if (
                type(quantity) is not int
                or not 1 <= quantity <= 20
                or not isinstance(selected, list)
                or not selected
            ):
                return {"ok": False, "error": "Enter a quantity and select product variants."}
            if any(not isinstance(x, str) for x in selected) or len(set(selected)) != len(selected):
                return {"ok": False, "error": "Product selection is invalid."}
            variants = {v["id"]: v for v in w["products"][key]["variants"]}
            if set(selected) - set(variants):
                return {"ok": False, "error": "Some product variants are no longer available."}
            prepared[key] = {
                "quantity": quantity,
                "selected": {x: variants[x]["color"] for x in selected},
            }
            value_sets.append(set(prepared[key]["selected"].values()))
        if combined and any(x != value_sets[0] for x in value_sets[1:]):
            return {"ok": False, "error": "Combined options must have matching names and values."}
        p["components"] = prepared
        p["combined"] = combined
        w["invalidated"] = []
        for row in variant_rows(w):
            p["prices"].setdefault(row["key"], row["sum"])
        return {"ok": True, "message": "Bundle saved."}
    if op == "save_product":
        status = c.get("status")
        online = c.get("online")
        prices = c.get("prices", {})
        if (
            not isinstance(status, str)
            or status not in {"active", "draft", "archived", "unlisted"}
            or type(online) is not bool
            or not isinstance(prices, dict)
        ):
            return {"ok": False, "error": "Product details are invalid."}
        keys = {row["key"] for row in variant_rows(w)}
        if set(prices) - keys:
            return {"ok": False, "error": "The variant list has changed."}
        try:
            validated = {key: money(value) for key, value in prices.items()}
        except (ValueError, TypeError):
            return {"ok": False, "error": "Enter a valid price."}
        # An unresolved child binding remains an invalid draft, as the source documents. Say so
        # instead of silently saving "Active" as Draft.
        if w["invalidated"] and status in {"active", "unlisted"}:
            return {
                "ok": False,
                "error": "This bundle still has changed product options and stays a draft.",
            }
        p["prices"].update(validated)
        p["online"] = online
        p["status"] = status
        return {"ok": True, "message": "Product saved."}
    if op == "component_price":
        product = w["products"].get(c.get("product")) if isinstance(c.get("product"), str) else None
        if not product:
            return {"ok": False, "error": "Product not found."}
        variant = next((v for v in product["variants"] if v["id"] == c.get("variant")), None)
        if not variant:
            return {"ok": False, "error": "Variant not found."}
        try:
            price = money(c.get("price"))
        except (ValueError, TypeError):
            return {"ok": False, "error": "Enter a valid price."}
        variant["price"] = price
        product["revision"] += 1
        return {"ok": True, "message": "Product saved."}
    if op == "other_status":
        if not isinstance(c.get("status"), str) or c["status"] not in {
            "active",
            "draft",
            "archived",
            "unlisted",
        }:
            return {"ok": False, "error": "Invalid status."}
        w["other"]["status"] = c["status"]
        return {"ok": True, "message": "Product saved."}
    return {"ok": False, "error": "Unknown operation."}


def components_preserved(w):
    return all(
        v["price"] == w["catalog_prices"][v["id"]]
        for p in w["products"].values()
        for v in p["variants"]
    )


from realtime_tasks.web.rt_bundle_catalog_rebind.tests.evaluation import evaluate


def terminal_reason(w):
    # A promotion order that could not be placed, or was placed on other terms, decides the
    # episode; close it then instead of running a lost campaign silently to 16:04:45.
    for order in w["orders"]:
        if order["missed"]:
            return f"{order['label']}_order_missed"
        if not order["listing_correct"]:
            return f"{order['label']}_order_wrong_terms"
    return (
        "campaign_ended" if w["now"] >= CAMPAIGN_ENDS_AT and not evaluate(w)["resolved"] else None
    )
