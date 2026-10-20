# JavaScript Pong provenance

- Upstream: https://github.com/jakesgordon/javascript-pong at `ca3240536e4f79ab7144388e56ed19de715b6662`.
- License: MIT, preserved at `upstream/LICENSE`.
- Vendoring: `upstream/` is the unmodified GitHub source tree. Root files are the benchmark launch
  page (`play.html`), the iframe wrapper (`index.html`, `game_api.js`) and the read-only
  observation adapter (`game_config.js`).

## Benchmark configuration (launch page, upstream untouched)

`play.html` passes two values through the upstream configuration hook (`Game.start(id, game, cfg)`,
`upstream/game.js:194` extends `Pong.Defaults` with `cfg`):

| cfg key     | upstream default (`upstream/pong.js:14-15`) | benchmark value | effect |
|-------------|---------------------------------------------|-----------------|--------|
| `ballSpeed` | `4` (court crossed horizontally in 4 s; 157.5 px/s) | `8` (78.75 px/s, 8 s per crossing) | slower ball |
| `ballAccel` | `8` (px/s^2, ball speeds up continuously during a rally, `pong.js:574-579`) | `0` | no continuous acceleration; paddle spin can still change vertical velocity |

Rationale (2026-09-19): with the upstream values a rally shortens each crossing to 1-2 s within
20-30 s, so at one paddle move per second the task was a reaction test that a scripted reactive
follower and a scripted predictor both mostly lose on timing, not on spatial reasoning. At 8 s per
horizontal crossing and no continuous acceleration a scripted predictor (landing point from ball position and velocity, one
move per second) returns every ball, while parked, oscillating, random and current-y-following
paddles concede 3 points within 33-80 s (see the task PR for the tables).

Everything else is upstream: paddle speed (`paddleSpeed: 2`, 198 px/s), paddle height 60, computer
AI levels and rubber-banding (`pong.js:35-53, 102-104`), spin from an active paddle direction
(`pong.js:542-546`). After normal collision reflection, either paddle's active direction
multiplies vertical velocity by 0.5 when aligned with the ball's vertical direction or 1.5
when opposed; this multiplier does not reverse direction or change horizontal velocity.
No multiplier applies when neither direction is active. The direction flags remain active
at court boundaries. Holding both Q and A stops the left paddle's movement, but the up flag
(Q) takes priority for spin. These native rules are included in the agent-visible `task.toml`
rules; the upstream physics are unchanged.

## Adapter (`game_config.js`)

- `game_state.player_returns`: number of rebounds off the left paddle, derived read-only from the
  live ball. The ball's horizontal velocity flips from negative to positive only when it rebounds
  off the left paddle (`pong.js:525-534`); goals reset the ball without a sign flip
  (`pong.js:476-480`). Sampled by the wrapper's 250 ms poll and on every `getState`.
- Terminal: `opponent_reached_three` when the computer's score reaches 3. With continuous acceleration disabled
  the computer AI (re-predicts every 1.0 s at level 8 with error proportional to the ball's distance)
  does not miss, so the player cannot score points; the success metric is the return count.
- Game clock: the upstream loop is `setInterval(loop, 1000/60)` with `dt` from `Date.now()`
  (`upstream/game.js:214-229`, no requestAnimationFrame), so CDP virtual time freezes and advances
  it exactly (measured 0 px moved during a 3 s wall pause; 1000 ms of game time per 1 s advance).

## Source and asset audit

All 37 files under `upstream/` match the fixed commit above, including the PNG
images and WAV sounds. Their source paths and SHA-256 values are recorded in
`PROVENANCE.json`. The game repository's MIT notice is preserved; it does not
provide separate asset-author records. Benchmark changes are in the sibling launch
and observation files, as described above.
