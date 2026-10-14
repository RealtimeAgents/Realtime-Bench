from __future__ import annotations


def evaluate(world):
    from realtime_tasks.web.rt_dispatch_address_hold.environment.engine import (
        FIRST_COLLECTION_AT,
        _address_key,
        deepcopy,
    )

    dispatched = world["dispatch"]
    shipped = _address_key(dispatched["shipping"]) if dispatched else None
    address_ok = bool(dispatched) and shipped == _address_key(world["corrected"])
    superseded = bool(dispatched) and shipped == _address_key(world["confirmed"])
    resolved = bool(
        dispatched
        and dispatched["at"] == world["pickup_at"]
        and address_ok
        and (world["hold_at"] is not None)
        and (world["hold_at"] < FIRST_COLLECTION_AT)
        and (world["release_at"] is not None)
        and (world["release_at"] >= world["confirmation_at"])
        and (not world["violations"])
    )
    return {
        "resolved": resolved,
        "dispatched": bool(dispatched),
        "address_matches_correction": address_ok,
        "address_matches_first_confirmation": superseded,
        "violations": list(world["violations"]),
        "commands": deepcopy(world["commands"]),
    }
