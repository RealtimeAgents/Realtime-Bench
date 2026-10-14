from __future__ import annotations


def evaluate(w):
    from realtime_tasks.web.rt_floorplan_party_handoff.environment.engine import _party, _safe

    seated = all((_party(w, pid)["status"] == "Seated" for pid in ("casey", "ellis")))
    preserved = all(
        (
            _party(w, pid)["status"] == "Expected"
            and _safe(w, _party(w, pid), _party(w, pid)["tables"], w["now"])
            for pid in ("nina", "brooks")
        )
    )
    return dict(
        resolved=bool(seated and preserved and (not w["violations"])),
        seated_waitlist_parties=sum(
            (_party(w, pid)["status"] == "Seated" for pid in ("casey", "ellis"))
        ),
        future_reservations_preserved=preserved,
        unsafe_actions=len(w["violations"]),
    )
