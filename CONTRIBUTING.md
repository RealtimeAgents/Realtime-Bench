# Contributing

- **Scope.** Use a focused branch. Do not change task objectives, evaluation timing or scoring
  unless the change is about them, and say so in the pull request.
- **Where code goes.** Framework code lives in `realtime_framework/` (entity, harnesses,
  providers, evaluation, sandbox). Task logic, assets and scorers live in their task package;
  suite membership only in `realtime_tasks/sets/`. New model APIs: [models](docs/models.md#adding-a-model-api).
  New tasks: [adding tasks](docs/adding-tasks.md).
- **Comparability.** A change that alters results must change the suite or harness hash and
  bump the version ([maintainers](docs/maintainers.md#versions)); never edit results to match.
- **Vendored code.** Record its upstream URL, immutable revision, local changes, license text
  and asset origins in its task package.
- **Paths.** File and directory paths must stay unique when compared without case, so macOS
  checkouts produce the same suite hashes as Linux.
- **Public tasks only.** This repository holds `realtime_tasks/sets/public.yaml` and its tasks.
  Held-out tasks belong to the private repository.

CI runs the *Portable paths* and *Public tasks* checks. Fill in the pull request template.
