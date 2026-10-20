from __future__ import annotations


def evaluate(w):
    return dict(
        resolved=bool(
            w["boarded"] and w["first_met"] and (not w["additional_rides"]) and (not w["cancelled"])
        ),
        first_reply_on_time=bool(w["first_met"]),
        current_driver_confirmed=bool(w["final_snapshot"] and w["final_snapshot"]["ready"]),
        rider_picked_up=w["boarded"],
        additional_rides=len(w["additional_rides"]),
        cancelled=w["cancelled"],
    )
