"""The ``rtbench`` command: Harbor's CLI with Realtime-Bench setup, reports and viewer.

Harbor 0.23.0 has no viewer plugin hook; adapt the in-process app factory without
changing Harbor's installed Python modules or frontend assets. Do not register a
competing ``harbor`` console script in this project's package metadata.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
AGENT = "realtime_framework.evaluation.harbor:RealtimeBenchAgent"
AGENT_NAME = "Realtime-Bench Agent"
# Names recorded before the display name; still accepted in job files and with -a.
AGENT_ALIASES = {"rtbench", "realtime-bench"}
ENVIRONMENT = "realtime_framework.evaluation.harbor:ApptainerEnvironment"
DEFAULT_SUITE = "public"
# The token and pause clocks score duration_minutes of game time, which can take far longer in
# real time (up to ~17x in the reference runs). Harbor's 900 s task timeout would kill such
# trials unscored, so their agent timeout covers this many real seconds per scored second, plus
# the task's setup budget. A dead provider is still caught by the runner's watchdog.
TOKEN_CLOCK_REAL_TIME_FACTOR = 36
TASK_AGENT_TIMEOUT_S = 900


def _benchmark_results(folder: Path) -> bool:
    if folder in (ROOT / "temp/harbor-jobs", ROOT / "results/harbor-jobs"):
        return True
    if next(folder.glob("*/*/agent/benchmark"), None) is not None:
        return True
    # A new job can be viewed before it has produced its first trial artifact.
    for path in folder.glob("*/config.json"):
        try:
            config = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        agents = config.get("agents")
        if not isinstance(agents, list):
            continue
        if any(
            isinstance(agent, dict) and agent.get("import_path") == AGENT
            for agent in agents
        ):
            return True
    return False


def setup() -> None:
    """Build or reuse both task images."""
    from realtime_framework.sandbox.apptainer import build_image

    for kind in ("sandbox", "realtime"):
        print(f"Task image ready: {build_image(kind)}", flush=True)


def suite_tasks(suite: str) -> dict[str, str]:
    """Return ``realtime_tasks/sets/<suite>.yaml`` task ids, in order, with their category.

    ``suite`` may pin a release as ``NAME@MAJOR.MINOR``; another checked-out version is rejected.
    """
    from realtime_tasks.registry import suite_path

    manifest = yaml.safe_load(suite_path(ROOT, suite).read_text())
    return {
        task: group["public_category"]
        for group in manifest["categories"]
        for task in group["task_ids"]
    }


def suites() -> None:
    """Print each suite's and the harness's version; ``+dev`` means the hash differs from its lock.

    To release, set every manifest's ``version`` and ``suite_hash`` and pyproject.toml
    ``[tool.rtbench.harness]`` ``version`` and ``hash`` to the printed values.
    """
    from realtime_framework.evaluation.config import harness_fingerprint, harness_version
    from realtime_tasks.registry import ALL_TASKS, load_suite

    harness = harness_fingerprint()
    print(f"harness@{harness_version(harness)}\t\t{harness}")

    for path in sorted((ROOT / "realtime_tasks/sets").glob("*.yaml")):
        suite = load_suite(path.stem, ALL_TASKS)
        print(f"{suite.suite_id}@{suite.version}\t{suite.expected_total} tasks\t{suite.hash}")


def suite_dataset(suite: str, selected: list[str]) -> dict:
    """One Harbor dataset for the selected suite tasks, named for what it covers.

    Harbor names a local dataset after its folder and has no other label, so each name is a
    folder of links to the suite's tasks: ``Full (N)``, ``web (N)``, ``game (N)`` or
    ``partial (N)``. The job and its trials record the suite itself.
    """
    categories = suite_tasks(suite)
    selected = [task for task in categories if task in set(selected)]
    covered = {categories[task] for task in selected}
    if len(selected) == len(categories):
        label = "Full"
    elif len(covered) == 1 and len(selected) == list(categories.values()).count(*covered):
        label = next(iter(covered))
    else:
        label = "partial"
    from realtime_tasks.registry import parse_suite

    folder = ROOT / "temp/harbor-suites" / parse_suite(suite)[0] / f"{label} ({len(selected)})"
    folder.mkdir(parents=True, exist_ok=True)
    for task in categories:
        sources = [
            ROOT / "realtime_tasks" / family / task
            for family in ("game", "web")
            if (ROOT / "realtime_tasks" / family / task / "task.toml").is_file()
        ]
        if len(sources) != 1:
            raise SystemExit(f"Suite {suite} names {task} without one task.toml in realtime_tasks/")
        link = folder / task
        if not (link.is_symlink() and link.resolve() == sources[0]):
            link.unlink(missing_ok=True)
            link.symlink_to(sources[0], target_is_directory=True)
    for entry in folder.iterdir():
        if entry.name not in categories:
            entry.unlink()
    return {"path": str(folder), "task_names": selected}


def suite_datasets(suite: str, patterns: list[str]) -> list[dict]:
    """Select tasks from ``realtime_tasks/sets/<suite>.yaml`` as one Harbor dataset."""
    task_ids = list(suite_tasks(suite))
    for pattern in patterns:
        if not any(fnmatchcase(task, pattern) for task in task_ids):
            raise SystemExit(f"No task in suite {suite} matches {pattern!r}; choose --suite")
    selected = [
        task for task in task_ids if not patterns or any(fnmatchcase(task, p) for p in patterns)
    ]
    return [suite_dataset(suite, selected)]


def family_datasets(datasets: list[dict], suite: str) -> list[dict]:
    """Replace job-file datasets naming ``realtime_tasks/game`` or ``/web`` with the suite."""
    families = {ROOT / "realtime_tasks" / family for family in ("game", "web")}
    paths = [(Path.cwd() / str(d.get("path", ""))).resolve() for d in datasets]
    if not datasets or any(path not in families for path in paths):
        return datasets
    task_ids = list(suite_tasks(suite))
    selected = set()
    for dataset, path in zip(datasets, paths):
        names = dataset.get("task_names")
        if names is None:
            names = [task for task in task_ids if (path / task / "task.toml").is_file()]
        missing = sorted(set(names) - set(task_ids))
        if missing:
            raise SystemExit(f"{missing} are not in suite {suite}; set the agent's suite")
        selected.update(names)
    return [suite_dataset(suite, sorted(selected))]


def describe_agent(agent: dict) -> None:
    """Give Harbor the agent name and ``vendor/model`` it displays and groups results by.

    Harbor still imports the class from ``import_path``. A catalog preset keeps its key as the
    model name, qualified by the model's vendor, and the agent receives the key unchanged.
    """
    import realtime_framework.providers.registry as catalog

    agent["name"] = AGENT_NAME
    kwargs = agent.setdefault("kwargs", {})
    model = kwargs.get("catalog_model") or agent.get("model_name")
    if model and "/" not in model and not (kwargs.get("base_url") or kwargs.get("api_base")):
        if catalog.has(model):
            agent["model_name"] = f"{catalog.vendor(model)}/{model}"
            kwargs["catalog_model"] = model


def _take(arguments: list[str], *flags: str) -> list[str]:
    """Remove every ``flag value`` / ``flag=value`` occurrence and return the values."""
    values, rest, index = [], [], 0
    while index < len(arguments):
        argument = arguments[index]
        name, equals, value = argument.partition("=")
        if argument in flags:
            if index + 1 >= len(arguments):
                raise SystemExit(f"{argument} needs a value")
            values.append(arguments[index + 1])
            index += 2
            continue
        if equals and name in flags and name.startswith("--"):
            values.append(value)
        else:
            rest.append(argument)
        index += 1
    arguments[:] = rest
    return values


def plan_job(path: Path) -> dict:
    """Translate a ``configs/runs/`` evaluation plan into a Harbor job configuration.

    Harbor ignores keys it does not know, so a plan passed as a job file would otherwise
    run every model on the default wall clock.
    """
    from realtime_framework.evaluation.config import load_benchmark

    try:
        plan = load_benchmark(path)
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from exc
    if plan["cells"] or any("endpoint" in model for model in plan["models"]):
        raise SystemExit(f"{path}: rtbench runs plan models only by configs/models/ name")
    runtime = plan["runtime"]
    kwargs = {
        "suite": plan["suite"],
        "fps": runtime["fps"],
        "model_input_fps": runtime["model_input_fps"],
        "interaction_mode": runtime["interaction_mode"],
        "duration_minutes": runtime["duration_minutes"],
        "seed": runtime["seed"],
        "clock": runtime["clock_mode"],
    }
    if runtime["clock_mode"] == "token":
        kwargs["tokens_per_frame"] = runtime["tokens_per_frame"]
    return {
        "agents": [
            {
                "import_path": AGENT,
                "model_name": model["key"],
                "kwargs": {**kwargs, "reasoning_effort": model["reasoning_effort"]},
            }
            for model in plan["models"]
        ],
        "datasets": [suite_dataset(plan["suite"], plan["tasks"])],
    }


def run_arguments(arguments: list[str]) -> list[str]:
    """Translate ``rtbench run`` options into a complete Harbor job configuration.

    ``-c`` may be any Harbor job YAML; benchmark defaults fill what it omits. ``--suite`` and
    ``-i`` select tasks from ``realtime_tasks/sets/`` and replace the configured datasets.
    All other options are passed to Harbor unchanged.
    """
    arguments = list(arguments)
    hub = sorted({a.split("=", 1)[0] for a in arguments} & HUB_OPTIONS)
    if hub:
        raise SystemExit(
            f"{', '.join(hub)}: Realtime-Bench results are not uploaded to Harbor Hub; "
            "submit them with `uv run rtbench upload` (docs/submit.md)"
        )
    configs = _take(arguments, "-c", "--config")
    models = _take(arguments, "-m", "--model")
    agents = _take(arguments, "-a", "--agent")
    suites = _take(arguments, "--suite")
    patterns = _take(arguments, "-i", "--include-task-name")
    if len(configs) > 1 or len(suites) > 1:
        raise SystemExit("Give -c and --suite at most once")
    if any(agent not in {*AGENT_ALIASES, AGENT_NAME, AGENT} for agent in agents):
        raise SystemExit("Realtime-Bench runs its own agent; omit -a or use -a rtbench")

    config = (yaml.safe_load(Path(configs[0]).read_text()) or {}) if configs else {}
    plan = bool({"runtime", "models"} & set(config))
    if plan:
        config = plan_job(Path(configs[0]))
    config.setdefault("jobs_dir", "./temp/harbor-jobs")
    config.setdefault("n_concurrent_trials", 1)
    config.setdefault("environment", {"import_path": ENVIRONMENT, "kwargs": {"lane": 1}})
    base_agents = config.get("agents") or [{"import_path": AGENT}]
    if models:
        if plan:
            # -m replaces the plan's models; their efforts may not apply to it.
            base_agents = [
                {**base_agents[0], "kwargs": {
                    k: v for k, v in base_agents[0]["kwargs"].items() if k != "reasoning_effort"
                }}
            ]
        base_agents = [
            {**agent, "kwargs": dict(agent.get("kwargs") or {}), "model_name": model}
            for agent in base_agents
            for model in models
        ]
    from harbor.cli.utils import parse_kwargs

    # Harbor merges --ak into every agent after this point; read them without consuming them.
    cli_kwargs = parse_kwargs(_take(list(arguments), "--ak", "--agent-kwarg"))
    for agent in base_agents:
        if agent.get("name") in {None, AGENT_NAME, *AGENT_ALIASES}:
            agent.setdefault("import_path", AGENT)
        if agent.get("import_path") == AGENT:
            describe_agent(agent)
            kwargs = {**(agent.get("kwargs") or {}), **cli_kwargs}
            if kwargs.get("clock") == "token":
                scored_s = 60 * float(kwargs.get("duration_minutes", 5.0))
                agent.setdefault(
                    "override_timeout_sec",
                    TASK_AGENT_TIMEOUT_S + TOKEN_CLOCK_REAL_TIME_FACTOR * scored_s,
                )
        if suites:
            agent.setdefault("kwargs", {})["suite"] = suites[0]
    config["agents"] = base_agents
    if suites or patterns or not (config.get("datasets") or config.get("tasks")):
        suite = suites[0] if suites else DEFAULT_SUITE
        config["datasets"] = suite_datasets(suite, patterns)
        config["tasks"] = []
        for agent in config["agents"]:
            agent.setdefault("kwargs", {}).setdefault("suite", suite)
    elif config.get("datasets"):
        suite = next(
            (a["kwargs"]["suite"] for a in base_agents if (a.get("kwargs") or {}).get("suite")),
            DEFAULT_SUITE,
        )
        config["datasets"] = family_datasets(config["datasets"], suite)

    folder = ROOT / "temp/harbor-configs"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=folder, delete=False) as output:
        yaml.safe_dump(config, output, sort_keys=False)
    return ["-c", output.name, *arguments]


# Harbor Hub options of `harbor run`; results go to the benchmark's own server instead.
HUB_OPTIONS = {"--upload", "--launch", "--public", "--private", "--share-org", "--share-user", "--org"}


def _register_commands(app):
    import typer

    # Harbor's `upload` sends a job to Harbor Hub; `rtbench upload` submits to Realtime-Bench.
    hub_upload = [command for command in app.registered_commands if command.name == "upload"]
    for command in hub_upload:
        app.registered_commands.remove(command)
    for name in ("report", "setup", "suites"):
        if any(command.name == name for command in app.registered_commands):
            raise RuntimeError(
                f"Harbor already has a {name} command; update the benchmark integration"
            )
    registered = len(app.registered_commands)

    @app.command(name="setup", help="Build the task images.")
    def setup_command():
        setup()

    @app.command(name="suites", help="List task suites and the harness with their version and hash.")
    def suites_command():
        suites()

    @app.command(name="report", help="Validate and summarize a Realtime-Bench evaluation.")
    def report(
        job: Path = typer.Option(..., "--job", help="Harbor job directory to aggregate."),
        output: Path = typer.Option(..., "--output", help="Destination JSON report."),
        official: bool = typer.Option(False, "--official", help="Maintainer-only private report."),
    ):
        from realtime_framework.evaluation.harbor import harbor_report_main

        arguments = ["--job", str(job), "--output", str(output)]
        if official:
            arguments.append("--official")
        raise typer.Exit(harbor_report_main(arguments))

    @app.command(name="upload", help="Check three public runs and submit them to the leaderboard.")
    def upload_command(
        jobs: list[Path] = typer.Argument(..., help="The three job directories, one per run."),
        url: str = typer.Option(None, "--url", help="Submission server (default: RTBENCH_SUBMIT_URL)."),
        keep_archive: bool = typer.Option(False, "--keep-archive", help="Keep the uploaded archive."),
    ):
        from realtime_framework.evaluation.submission import SubmissionError, main as submit

        try:
            result = submit(jobs, url=url, keep_archive=keep_archive)
        except SubmissionError as exc:
            typer.echo(f"Not submitted: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"Submitted {result['id']}. Email realtime-agents@utlists.utexas.edu with this ID.")

    return app.registered_commands[registered:], hub_upload


def _rtbench_console_hints():
    """Point Harbor's closing hints at ``uv run rtbench``, whose viewer adds Recording and timing."""
    import importlib

    patched = []
    for name in ("harbor.cli.jobs", "harbor.cli.exec"):
        console = getattr(importlib.import_module(name), "console", None)
        if console is None:
            continue
        print_ = console.print

        def rewritten(*objects, _print=print_, **kwargs):
            objects = tuple(
                item.replace("`harbor view ", "`uv run rtbench view ")
                if isinstance(item, str)
                else item
                for item in objects
            )
            return _print(*objects, **kwargs)

        console.print = rewritten
        patched.append((console, print_))
    return patched


def main():
    # Harbor's telemetry would report each job's agent and model names; users opt in with 1.
    os.environ.setdefault("HARBOR_TELEMETRY", "0")
    from harbor.cli.main import app
    import harbor.viewer as viewer

    if sys.argv[1:2] == ["run"] and not {"--help", "-h"} & set(sys.argv):
        sys.argv[2:] = run_arguments(sys.argv[2:])
        # Builds are idempotent and fingerprinted, so every run starts from current images.
        if "--print-config" not in sys.argv:
            setup()
    registrations, hub_upload = _register_commands(app)
    original = viewer.create_app

    def create_app(folder, *, mode="jobs", static_dir=None):
        folder = Path(folder).expanduser().resolve()
        if mode != "jobs" or static_dir is None or not _benchmark_results(folder):
            return original(folder, mode=mode, static_dir=static_dir)

        # Initialize pricing before requests can import it in an async worker.
        import litellm  # noqa: F401

        from realtime_framework.evaluation.harbor_viewer import create_app as benchmark_app

        return benchmark_app(folder, static_dir=Path(static_dir))

    viewer.create_app = create_app
    consoles = _rtbench_console_hints()
    try:
        return app()
    finally:
        viewer.create_app = original
        for console, print_ in consoles:
            console.print = print_
        for registration in registrations:
            app.registered_commands.remove(registration)
        app.registered_commands.extend(hub_upload)
