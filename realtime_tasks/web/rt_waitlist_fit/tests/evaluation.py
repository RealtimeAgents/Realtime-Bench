from __future__ import annotations


def evaluate(w):
    bookings = [a for a in w["appointments"] if not a["original"] and a["status"] == "Confirmed"]
    resolved = (
        not w["violations"] and len(bookings) == 1 and (bookings[0]["notification"] == "Sent")
    )
    return {
        "resolved": bool(resolved),
        "active_new_appointments": len(bookings),
        "notifications_delivered": sum((a["notification"] == "Sent" for a in bookings)),
        "violations": list(w["violations"]),
    }
