# Submit results

Realtime-Bench has two task sets:

| Set | Tasks | Run by | Purpose |
| --- | --- | --- | --- |
| [Public](public_tasks.md) | 20 (12 Web, 8 Game) | You | The public benchmark; results go on the [leaderboard](https://realtimeagents.org) |
| Private | 30 (18 Web, 12 Game) | Us | A held-out set that is never released, so models cannot be tuned on it |

You run the public tasks and submit the results; we then run the private tasks for your model.

## Run the public tasks

Run the full public suite three times with the default settings:

```bash
for run in 1 2 3; do
  uv run rtbench run --suite public@1.0 -m "provider/model-id" --job-name my-model-run$run
done
```

Submit all three runs as they are. A trial that ends in `error` counts as a failure. The
leaderboard scores every task over the three runs (pass^k and pass@k).

## Submit to the leaderboard

1. **Get a submission token.** Email **realtime-agents@utlists.utexas.edu** with the model's name
   and your organization. Put the token we send you in `.env` as `RTBENCH_SUBMIT_TOKEN`.
2. **Upload the three runs:**

   ```bash
   uv run rtbench upload temp/harbor-jobs/my-model-run{1,2,3}
   ```

3. **Reply with the submission ID** the command prints, plus the model's `provider/model-id` or
   [preset](model_configuration.md#write-a-preset) YAML, its deployment revision and reasoning
   effort, and the name and organization to show on the leaderboard.

We review the runs and add the model to the leaderboard.

## Run the private tasks

We run the private tasks for you with the same harness, settings and three runs:

1. **Integrate your model and run the public tasks.** Set up your model as described in
   [model configuration](model_configuration.md), then [run the public tasks](#run-the-public-tasks)
   and [submit the results](#submit-to-the-leaderboard). Continue once these look right.
2. **Open a pull request with your model integration:** a preset in `configs/models/<name>.yaml`,
   or a new model API if your model needs one. Use the same integration that produced your public
   results.
3. **Email the API key or endpoint access** to **realtime-agents@utlists.utexas.edu**, with enough
   quota for 90 five-minute tasks. Keep keys out of the pull request.

We merge the integration, run the private tasks and add the results to the leaderboard.

## Contact

Questions: **realtime-agents@utlists.utexas.edu**.
