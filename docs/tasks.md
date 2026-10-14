# Public tasks

[public.yaml](../realtime_tasks/sets/public.yaml) defines the suite's membership and order;
`uv run rtbench run` selects it by default. The held-out set is maintained privately
([maintainers](maintainers.md#private-tasks)).

## Web

| Task | Objective |
| --- | --- |
| `rt_accessible_trip_dispatch` | Dispatch accessible rides and reconcile a no-show with its separate return trip. |
| `rt_floorplan_party_handoff` | Seat arriving parties while managing joined tables, live updates, and future reservations. |
| `rt_waitlist_fit` | Fit a waitlisted appointment around new bookings and changing client availability. |
| `rt_pickup_driver_handoff` | Update the structured pickup point when a replacement driver reports a different entrance. |
| `rt_document_revision_routing` | Withdraw an obsolete agreement, reconcile updates, and route fresh signatures in order. |
| `rt_return_receipt_exchange` | Reconcile received and rejected return items, inventory location, and paid exchange release. |
| `rt_identity_recovery_handoff` | Complete account recovery, activate the replacement device, and retire the correct old devices. |
| `rt_bundle_catalog_rebind` | Repair bundle variants after component changes, update prices, and publish valid inventory. |
| `rt_cruise_batch_closeout` | Close cancelled cruise sales and settle booking groups according to their separate consent. |
| `rt_shipment_regroup_handoff` | Correct shipment grouping and a late add-on before the carrier handoff. |
| `rt_dispatch_address_hold` | Hold fulfillment, apply confirmed address corrections, and resend the warehouse request in time. |
| `rt_room_move_maintenance` | Relocate guests and block damaged-room inventory as maintenance dates change. |

## Game

| Task | Objective |
| --- | --- |
| `gh_timed_chess` | Capture four Black pieces before the two-minute White clock expires. |
| `gomoku` | Connect five stones against the AI with ten seconds per move. |
| `gh_javascript_pong` | Return the ball six times before the computer scores three points. |
| `gh_synthblast` | Activate every objective pad in two arenas and reach Level 3 within 180 seconds. |
| `gh_norman_necromancer` | Aim spells and time resurrections to bring the soul counter to 200. |
| `gh_xx142_b2exe` | Coordinate a 13-second route with a replay ghost to open the timed door and escape. |
| `gh_evolve` | Allocate RNA and DNA to acquire four evolution upgrades. |
| `gh_highway_404` | Collect a teapot and survive 30 seconds on a collapsing highway. |
