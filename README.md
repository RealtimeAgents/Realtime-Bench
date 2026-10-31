# Realtime-Bench

[![Project Page](https://img.shields.io/badge/Project_Page-000000?style=for-the-badge&logo=googlechrome&logoColor=white)](https://realtimeagents.org/) [![Paper](https://img.shields.io/badge/arXiv-2610.22155-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.22155)

A benchmark for AI agents acting in real-time web and game environments. The world
keeps changing while the agent reasons, so success depends on both the action and
when it arrives.

Realtime-Bench 1.0 publishes [20 tasks](docs/public_tasks.md) (12 Web, 8 Game) and keeps a held-out
set private for official evaluation. Every task runs on the [wall clock](docs/evaluation.md#choose-a-clock)
at 1 FPS for five minutes.

## Quickstart

Requires Linux, [uv](https://docs.astral.sh/uv/), and Apptainer with fakeroot.

```bash
git clone https://github.com/RealtimeAgents/Realtime-Bench.git realtime-bench
cd realtime-bench
uv sync
cp .env.example .env   # then add your provider's API key
```

Run it from this checkout with `uv`. Installing it as a package, for example with `pip install .`,
is not supported.

Run a model by its file name in [`configs/models/`](configs/models), then view the results. The
first run builds the task images, which takes a while:

```bash
uv run rtbench run -m gemini-3.8-flash
uv run rtbench view ./temp/harbor-jobs --port 18081
```

To add a model, copy `configs/models/my-model.yaml`, fill in its connection and run it by its new
name. A model file fixes the route and settings, so use one for real experiments and new models.

`-m "provider/model-id"` also runs a LiteLLM model, on LiteLLM's defaults. It cannot set a file's
route, output limit or provider options, so its results can differ from the file's.

## Documentation

| Page | For |
| --- | --- |
| [Evaluation](docs/evaluation.md) | Connecting a model, running it, viewing results |
| [Reference](docs/reference.md) | Every run option, model setting and result field |
| [Custom drivers](docs/custom_drivers.md) | Adding a new model API or an external driver |
| [Public tasks](docs/public_tasks.md) | The 20 public tasks |
| [Submit](docs/submit.md) | Submitting to the leaderboard and evaluating on the private tasks |

To contribute a task, see [CONTRIBUTING](CONTRIBUTING.md).

## Citation


```bibtex
@misc{tang2026realtimebenchfrontiermodelssolve,
      title={Realtime-Bench: Can Frontier Models Solve Real-time Interactive Problems?},
      author={Jingyu Tang and Xu Yang and Zhehao Zhang and Zheng Wang and Qi Cheng and Haozhe Zhao and Manling Li and Wenxuan Li and Xide Xia and Jiaxin Pei},
      year={2026},
      eprint={2610.22155},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.22155},
}
```
