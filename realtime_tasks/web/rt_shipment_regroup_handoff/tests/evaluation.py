from __future__ import annotations


def evaluate(w):
    from realtime_tasks.web.rt_shipment_regroup_handoff.environment.engine import (
        HANDOFF_AT,
        SERVICES,
        _get,
        _live,
        _order,
    )

    orders = {o["id"]: o for o in w["orders"]}
    mara = {i: o for i, o in orders.items() if o["customer"] == "Mara Lin"}
    active = [l for l in w["labels"] if l["status"] == "Active"]
    quantities, owners = ({}, {})
    for s in _live(w):
        for i in s["items"]:
            quantities[i["order"]] = quantities.get(i["order"], 0) + i["quantity"]
            if i["order"] in mara:
                owners.setdefault(s["owner"], set()).add(mara[i["order"]]["requested"])
    shipped, correct = ({}, True)
    for label in active:
        if not label["items"] or label["created_s"] >= HANDOFF_AT:
            correct = False
        for i in label["items"]:
            order = mara.get(i["order"])
            if (
                not order
                or label["service"] != order["requested"]
                or label["address"] != order["address"]
            ):
                correct = False
            shipped[i["order"]] = shipped.get(i["order"], 0) + i["quantity"]
    original = all((orders.get(o["id"]) == o for o in w["original_orders"]))
    if w["addon_received"]:
        original = original and orders.get(w["addon_id"]) == _order(
            w["addon_id"], w["addon_quantity"], w["addon_service"], w["addon_at"]
        )
    checks = {
        "timely_old_label_void": _get(w["labels"], "L710001")["status"] == "Voided"
        and (not w["old_label_used"]),
        "all_orders_received": w["addon_received"],
        "original_orders_preserved": original,
        "quantity_conservation": quantities == {i: o["quantity"] for i, o in orders.items()},
        "separate_order_groups": all((len(values) == 1 for values in owners.values())),
        "all_items_labelled_once": shipped == {i: o["quantity"] for i, o in mara.items()},
        "two_service_parcels": len(active) == 2
        and {l["service"] for l in active} == set(SERVICES[:2]),
        "label_contents_and_service": correct,
        "unrelated_shipment_preserved": _get(w["shipments"], "S202") == w["unrelated_shipment"],
    }
    return {"resolved": all(checks.values()), "checks": checks}
