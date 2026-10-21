"""Differential runs of today's decision loop against the event kernel (phase2.md, 事件内核,
怎么验证; wave2-plan.md, P4).

One scripted world (script.py) is played three ways and projected into one shape (projection.py),
decision by decision: the kernel's discrete-event simulation (simulated.py), and a real agent on
the replay stubs and fake environments under the kernel's shell and under today's loop
(replayed.py, drivers.py). The shell and the simulation agree exactly; today's loop and the shell
differ only by the numbered expected differences (rules.py). The named scripts (scenarios.py,
test_named.py) and the random ones (generate.py, test_random.py) check that nothing else differs,
and the session summary (conftest.py) says which differences each script showed. REPORT.md is the
sign-off.

Today's loop is legacy_loop.py, the turn loop main() ran before the kernel's shell, moved here
unchanged (decision 11); drivers.legacy calls it.
"""
