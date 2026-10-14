from __future__ import annotations


def evaluate(w):
    from realtime_tasks.web.rt_identity_recovery_handoff.environment.engine import (
        ENDS_AT,
        RETIRE_BY,
        _phone,
        _recovered_in_time,
    )

    first, new = (w["_first_login"], w["_new_push"])
    current = _phone(w, w["_replacement"])
    shared = _phone(w, w["_shared"])
    retired = all(
        (
            _phone(w, p) is None and w["_retired"].get(p, ENDS_AT + 1) <= RETIRE_BY
            for p in [w["_old"], w["_spare"]]
        )
    )
    checks = dict(
        verified_timely_recovery=_recovered_in_time(first),
        confirmed_activation_delivery=any((d["authorized"] for d in w["_deliveries"])),
        actual_replacement_push=bool(new and new["at"] <= RETIRE_BY),
        personal_devices_retired=retired,
        replacement_and_shared_preserved=bool(
            current
            and current["mobile"] == "Activated"
            and (current["users"] == ["mlane"])
            and shared
            and (set(shared["users"]) == {"mlane", "rortiz"})
            and (shared["mobile"] == "Activated")
        ),
        safe_recovery_history=not w["_violations"],
        mfa_status_active=w["user"]["status"] == "Active",
    )
    return dict(resolved=all(checks.values()), checks=checks)
