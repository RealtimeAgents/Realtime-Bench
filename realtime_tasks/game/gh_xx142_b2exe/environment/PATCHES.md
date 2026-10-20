# Task-local integration patches

The runnable files under `upstream/` are copied from the pinned source archive, with three small
integration changes:

1. `game.js` adds `getBenchmarkState()`, a read-only snapshot of the native level, timer, replay
   ghosts, switch/door state, and player position.
2. `main.js` exposes that snapshot through `window.xx142BenchmarkState` for the parent adapter.
3. `styles.css` corrects the upstream source selector from `#C` to the actual lowercase canvas ID
   `#c`, matching the shipped HTML so the canvas is visible after the title screen.

The wrapper starts the native title sequence, records trusted player input, and reports the
read-only snapshot through `gameAPI`. It does not alter the 13-second timer, movement, collision,
switch behavior, ghost replay, level geometry, or win conditions.
