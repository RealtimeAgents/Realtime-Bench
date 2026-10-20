## Summary

<!-- What changes and why. -->

## Validation

<!-- Commands run and their results. -->

## Checklist

- [ ] No change to task content, scoring, timing or harness behavior, **or** `uv run rtbench suites`
      output is attached.

New or changed task (delete otherwise):

- [ ] Meets RT-Perceive, RT-Deliberate and RT-Act ([task guidelines](../CONTRIBUTING.md#requirements)),
      and reflects a real need rather than a contrived setup.
- [ ] `instruction.md` and `tests/evaluation.py` agree: everything scored is stated.
- [ ] Doing nothing fails; a different seed changes the content.
- [ ] Success cannot be reached through browser chrome, URL changes or hidden state.
- [ ] Preview checked (reset, clock transitions, terminal conditions); one strong-model run
      explained above.
- [ ] Games: upstream URL, revision, local changes and license recorded.
