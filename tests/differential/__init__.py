"""Differential runs of today's decision loop against the event kernel (phase2.md, 事件内核,
怎么验证; wave2-plan.md, P4).

One scripted world (script.py) is played by each side and projected into one shape
(projection.py), decision by decision: the kernel's discrete-event simulation (simulated.py), and a
real agent on the replay stubs and fake environments under a loop (replayed.py, drivers.py).
Today's loop and the kernel differ only by the numbered expected differences (rules.py); the
named scripts (scenarios.py, test_named.py) and the random ones (generate.py, test_random.py) check
that nothing else differs, and the session summary (conftest.py) says which differences each
script showed.

The production shell's run_episode joins as a second driver once it exists (P4b). When today's
loop is archived after sign-off, it moves here unchanged as legacy_loop.py, and drivers.legacy
calls it from there.
"""
