# Realtime-Bench

[![Project Page](https://img.shields.io/badge/Project_Page-000000?style=for-the-badge&logo=googlechrome&logoColor=white)](https://realtimeagents.org/) [![Paper](https://img.shields.io/badge/arXiv-2610.22155-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.22155)

A benchmark for AI agents acting in real-time web and game environments. The world
keeps changing while the agent reasons, so success depends on both the action and
when it arrives.

Realtime-Bench 1.0 publishes [20 tasks](docs/public_tasks.md) (12 Web, 8 Game) and keeps a held-out
set private for official evaluation. Every task runs on the [wall clock](docs/evaluation.md#clocks)
at 1 FPS for five minutes.

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
| [Evaluation](docs/evaluation.md) | Run options, clocks, job files, viewing results |
| [Model configuration](docs/model_configuration.md) | Configuring a new model |
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
