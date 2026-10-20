from __future__ import annotations


def evaluate(w):
    from realtime_tasks.web.rt_room_move_maintenance.environment.engine import TODAY, booking

    b = booking(w, "6108")
    split_ok = (
        len(b["segments"]) == 2
        and b["segments"][0] == {"room": "203", "start": "2026-09-16", "end": TODAY}
        and (b["segments"][1]["start"] == TODAY)
        and (b["segments"][1]["end"] == "2026-09-20")
        and (b["segments"][1]["room"] != "203")
    )
    cursor = TODAY
    block_ok = bool(w["blocks"])
    for block in sorted(w["blocks"], key=lambda x: x["start"]):
        if block["room"] != "203" or block["start"] != cursor or (not block["reason"].strip()):
            block_ok = False
            break
        cursor = block["end"]
    block_ok = block_ok and cursor == w["confirmed_end"]
    rehoused = all(
        (
            len(p["segments"]) == 1
            and p["segments"][0]["room"] != "203"
            and (
                (p["segments"][0]["start"], p["segments"][0]["end"])
                == (p["arrival"], p["departure"])
            )
            for p in (booking(w, i) for i in w["displaced"])
        )
    )
    return {
        "resolved": bool(
            w["revised"]
            and split_ok
            and block_ok
            and rehoused
            and (not w["violations"])
            and (not w["failed"])
        ),
        "stay_preserved": split_ok,
        "repair_inventory_blocked": block_ok,
        "arrival_rehoused": rehoused,
        "moved_at": w["moved_at"],
        "violations": list(w["violations"]),
        "failure": w["failed"],
    }
