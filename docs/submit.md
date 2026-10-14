# Submit results

Realtime-Bench has two task sets:

| Set | Tasks | Run by | Purpose |
| --- | --- | --- | --- |
| [Public](public_tasks.md) | 20 (12 Web, 8 Game) | You | The public benchmark; results go on the [leaderboard](https://realtimeagents.org) |
| Private | 30 (18 Web, 12 Game) | Us | A held-out set that is never released, so models cannot be tuned on it |

Submit public results first; we then run the private set for your model.

## Run the public tasks

Use an unmodified checkout of a release: `uv run rtbench suites` must show `public@1.0` and
`harness@1.0` without `+dev`. Keep the default settings (wall clock, 1 FPS, five minutes per task,
seed 42) and run the full suite **three times**, one job per run:

```bash
for run in 1 2 3; do
  uv run rtbench run --suite public@1.0 -m "provider/model-id" --job-name my-model-run$run
done
```

Each job must cover all 20 tasks. A trial that ends in `error` (provider or infrastructure
failure) counts as a failure; do not drop it or replace it with a better run. The leaderboard
scores every task over the three runs (pass^k and pass@k).

## Submit to the leaderboard

1. **Get a submission token.** Email **jingyu@utexas.edu** with the model's name and your
   organization; we reply with a token. Put it in `.env` as `RTBENCH_SUBMIT_TOKEN`.
2. **Upload the three runs.**

   ```bash
   uv run rtbench upload temp/harbor-jobs/my-model-run{1,2,3}
   ```

   The command first checks the runs (release versions, default settings, all tasks, one model
   configuration), then uploads them with their trajectories and recordings to our server. An
   interrupted upload resumes when you run the command again. Results are never sent to Harbor
   Hub. Without a token it only checks the runs.
3. **Reply with the submission ID** the command prints, plus the model's `provider/model-id` or
   [preset](model_configuration.md#write-a-preset) YAML, its deployment revision and reasoning
   effort, and the name and organization to show on the leaderboard.

We review the runs and add the model to the leaderboard.

## Run the private tasks

The private tasks are not distributed, so we run them for you with the same harness, settings and
three runs. After submitting public results:

1. **Open a pull request that integrates your model**, as described in
   [model configuration](model_configuration.md): a preset in `configs/models/<name>.yaml`, or a
   new model API if your model needs one. We run the private tasks with exactly this integration,
   so it must be the one that produced your public results.
2. **Send credentials privately.** Never put keys in the pull request. Email the API key or
   endpoint access to **jingyu@utexas.edu**, with enough quota for 90 five-minute tasks.

We merge the integration, run the private tasks and add the results to the leaderboard.

## Contact

Questions about running or submitting: **jingyu@utexas.edu**.
