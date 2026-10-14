# gomoku

| File | Contents |
| --- | --- |
| `task.toml` | `description` (summary), `rules`, `keys`, `score_field` and `target` |
| `instruction.md` | The model's goal |
| `tests/evaluation.py` | Success criterion |
| `RIGHTS.md` | Rights review |

The upstream **vajvip/gomoku** (single file, with a heuristic AI) lives in `environment/game/`. `index.html` and `game_api.js` configure Easy PVE with the AI as White and expose the same read-only state contract as the other games.

Preview: run `uv run python -m realtime_tasks.preview` from the repository root, open
`http://127.0.0.1:18080/`, and select this task.
