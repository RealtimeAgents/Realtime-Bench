# Realtime-Bench

A benchmark for AI agents acting in real-time Web and Game environments. The world
keeps changing while the agent reasons, so success depends on both the action and
when it arrives.

Realtime-Bench 1.0 publishes [20 tasks](docs/tasks.md) (12 Web, 8 Game) and keeps a held-out
set private for official evaluation. Every task runs under the [default protocol](docs/protocol.md):
wall clock, 1 FPS, five minutes.

## Quickstart

Requires Linux, [uv](https://docs.astral.sh/uv/), and Apptainer with fakeroot.

```bash
git clone https://github.com/RealtimeAgents/Realtime-Bench.git realtime-bench
cd realtime-bench
uv sync
cp .env.example .env   # then add your provider's API key
```

Run any screenshot-capable model by its LiteLLM `provider/model-id`, then view the results. The
first run builds the task images, which takes a while:

```bash
uv run rtbench run -m "provider/model-id"
uv run rtbench view ./temp/harbor-jobs --port 18081
```

## Documentation

| Page | For |
| --- | --- |
| [Running](docs/running.md) | Run options, job files, viewing results, parallel lanes |
| [Models](docs/models.md) | Model presets and adding a model API |
| [Protocol](docs/protocol.md) | Clocks, actions, outcomes, and what to report |
| [Public tasks](docs/tasks.md) | The 20 public tasks |
| [Adding tasks](docs/adding-tasks.md) | Writing a task package, step by step |
| [Experimental models](docs/experimental-models.md) | Choice models (GPT-6 Luna Decisions, JEV) |
| [Maintainers](docs/maintainers.md) | Releases, pinned images, seeding internals |

Contributions: see [CONTRIBUTING](CONTRIBUTING.md).
