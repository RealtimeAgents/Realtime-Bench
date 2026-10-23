from __future__ import annotations

# Harbor runtime
"""Runner-side metadata bridge for the shared benchmark and Harbor environment."""

import argparse as argparse
import json as json
import os as os
import sys as sys

from realtime_framework.entity.config import DEFAULT_TASK_SEED, task_seed


SUITE_SCHEMA = "realtime-bench-suite-v1"


def describe(suite_id):
    os.environ["TASK_SUITE"] = suite_id
    from realtime_framework.sandbox.apptainer import image_path
    from realtime_framework.evaluation.config import harness_fingerprint, harness_version
    from realtime_tasks.registry import ACTIVE_SUITE
    from realtime_tasks.registry import TASKS

    return {
        "schema": SUITE_SCHEMA,
        "suite_id": ACTIVE_SUITE.suite_id,
        "suite_version": ACTIVE_SUITE.version,
        "suite_hash": ACTIVE_SUITE.hash,
        "harness_version": harness_version(harness := harness_fingerprint()),
        "harness_hash": harness,
        "images": {kind: str(image_path(kind)) for kind in ("sandbox", "realtime")},
        "tasks": [
            {
                "task_id": task_id,
                "category": ACTIVE_SUITE.public_category(task_id),
                "instruction": TASKS[task_id].description(),
            }
            for task_id in ACTIVE_SUITE.task_ids
        ],
    }


def describe_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="public", help="NAME or NAME@MAJOR.MINOR")
    args = parser.parse_args()
    print(json.dumps(describe(args.suite)))


# Suite metadata
"""Read the installed suite identity in a fresh interpreter, so it reflects the current checkout."""

import subprocess as subprocess
from pathlib import Path as Path


def runtime(root, python, suite):
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "from realtime_framework.evaluation.harbor import describe_main; describe_main()",
            "--suite",
            suite,
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(completed.stdout)
    if metadata.get("schema") != SUITE_SCHEMA:
        raise ValueError("Unsupported suite metadata schema")
    return metadata


# Verifier
"""Grade a harness-produced local report.

Only a verified, solved episode earns reward 1. Every other outcome, including any error, earns 0;
the recorded outcome and error reason keep errors distinguishable from model failures.
"""


def verify(report, expected, request):
    if request.get("task") != expected:
        raise ValueError("Trial identity mismatch")
    integration = report.get("integration", {})
    if any(
        integration.get(key) != request[key] for key in ("model", "model_revision", "mode")
    ) or not integration.get("configuration_sha256"):
        raise ValueError("Report integration identity mismatch")
    if (request.get("catalog_model") or request.get("configuration_sha256")) and (
        not request.get("configuration_sha256")
        or integration["configuration_sha256"] != request["configuration_sha256"]
    ):
        raise ValueError("Model configuration changed after resolution")
    groups = report.get("groups", [])
    rows = [row for group in groups for row in group.get("tasks", [])]
    if (
        len(groups) != 1
        or len(rows) != 1
        or rows[0].get("status") != "scored"
        or rows[0].get("task_id", "").split("__", 1)[-1] != expected["task_id"]
        or type(rows[0].get("solved")) is not bool
        or rows[0].get("outcome") not in ("pass", "fail", "timeout")
        or (rows[0]["outcome"] == "pass") is not rows[0]["solved"]
    ):
        raise ValueError("Missing, duplicate, invalid or unscored task")
    group = groups[0]
    for field in (
        "suite_id", "suite_version", "suite_hash", "harness_hash", "fps", "model_input_fps",
        "interaction_mode", "seed",
    ):
        if group.get(field) != expected.get(field):
            raise ValueError(f"Report {field} mismatch")
    if (
        group.get("public_category") != expected["category"]
        or group.get("mode") != request["mode"]
        or group.get("protocol")
        != ("time-aware-token-v1" if request.get("clock", "wall") == "token"
            else "wall-clock")
        or group.get("clock_mode", "wall") != request.get("clock", "wall")
        or group.get("tokens_per_frame") != (
            request.get("tokens_per_frame")
            if request.get("clock") == "token" and request.get("tokens_per_frame") != "unlimited"
            else None
        )
        or group.get("cap") != (request.get("duration_minutes") or expected["duration_minutes"])
        or group.get("expected") != 1
        or group.get("scored") != 1
        or group.get("invalid") != 0
        or group.get("missing") != 0
        or group.get("partial") is not False
        or not group.get("runtime_config_hash")
    ):
        raise ValueError("Report runtime or completeness mismatch")
    return {"reward": float(rows[0]["solved"])}


def verifier_source():
    import inspect

    return (
        "import json\nfrom pathlib import Path\n\n"
        + inspect.getsource(verify)
        + "\n"
        + inspect.getsource(verifier_main)
        + "\nif __name__ == '__main__':\n    verifier_main()\n"
    )


def verifier_main():
    output = Path("/logs/verifier/reward.json")
    details = output.with_name("evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    details.unlink(missing_ok=True)
    # Scoring details live with the verifier output; the reward alone stays authoritative.
    # outcome is pass, fail, timeout or error; only a verified pass earns reward 1.
    record = {
        "reward": None,
        "outcome": "error",
        "error_category": "infrastructure",
        "error_reason": "missing_report",
        "error": None,
        "evaluation": None,
    }
    try:
        record["evaluation"] = json.loads(
            Path("/logs/agent/benchmark/evaluation.json").read_text()
        ).get("evaluation")
    except (OSError, ValueError, AttributeError):
        pass
    try:
        report = json.loads(Path("/workspace/report.json").read_text())
        rows = [row for group in report.get("groups", []) for row in group.get("tasks", [])]
        if len(rows) == 1:
            for key in ("outcome", "error_category", "error_reason"):
                record[key] = rows[0].get(key)
        expected = json.loads(Path("/tests/task.json").read_text())
        request = json.loads(Path("/workspace/request.json").read_text())
        reward = verify(report, expected, request)
    except Exception as exc:
        # An error is scored as a failure.
        record["error"] = f"{type(exc).__name__}: {exc}"
        if record["outcome"] != "error":
            record.update(
                outcome="error",
                error_category="evaluator",
                error_reason="report_verification_failed",
            )
        reward = {"reward": 0.0}
    record["reward"] = reward["reward"]
    details.write_text(json.dumps(record, indent=2) + "\n")
    output.write_text(json.dumps(reward) + "\n")
    reason = f" ({record['error_reason']})" if record["outcome"] == "error" else ""
    print(f"{record['outcome']}{reason}: reward={reward['reward']} (details: evaluation.json)")

# Agent
"""Harbor bridge to the local runner for registered model endpoints."""

from typing import Literal as Literal

from harbor.agents.base import BaseAgent as BaseAgent
from realtime_framework.evaluation.harbor_viewer import export_episode, populate_context
from harbor.agents.options import AgentOptions as AgentOptions
from pydantic import ConfigDict
from pydantic import Field as Field
from pydantic import model_validator as model_validator


class RealtimeBenchOptions(AgentOptions):
    model_config = ConfigDict(extra="forbid")

    suite: str = Field(
        default="public", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*(@[0-9]+\.[0-9]+)?$"
    )
    fps: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    model_input_fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    interaction_mode: Literal["turn_based", "streaming"] = "turn_based"
    duration_minutes: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    seed: int = DEFAULT_TASK_SEED
    model_revision: str | None = Field(default=None, min_length=1)
    catalog_model: str | None = Field(default=None, min_length=1)
    base_url: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    clock: Literal["wall", "token"] = "wall"
    tokens_per_frame: int | Literal["unlimited"] = "unlimited"
    stream: bool = True
    reasoning_effort: str | None = None

    @model_validator(mode="after")
    def validate_integration(self):
        from realtime_framework.providers.preflight import check_frame_rates

        self.seed = task_seed(self.seed)
        if self.model_input_fps is None:
            self.model_input_fps = self.fps
        check_frame_rates(self.fps, self.model_input_fps)
        if self.api_base and self.base_url and self.api_base != self.base_url:
            raise ValueError("Supply only one of api_base or base_url")
        self.base_url = self.api_base or self.base_url
        if self.catalog_model and (
            self.base_url or self.api_key_env or self.model_revision or not self.stream
        ):
            raise ValueError(
                "base_url, api_key_env, model_revision and stream apply to LiteLLM models; "
                "set a preset's connection in configs/models/"
            )
        if self.tokens_per_frame != "unlimited" and self.tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive or 'unlimited'")
        return self


class RealtimeBenchAgent(BaseAgent):
    options_model = RealtimeBenchOptions

    @staticmethod
    def name():
        return "Realtime-Bench Agent"

    def version(self):
        return "0.4.0"

    async def setup(self, environment):
        if not isinstance(environment, ApptainerEnvironment):
            raise TypeError("Use -e realtime_framework.evaluation.harbor:ApptainerEnvironment")
        if not self.model_name:
            raise ValueError("Supply a LiteLLM provider/model-id or catalog preset with -m")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, instruction, environment, context):
        # mode is a fixed identity field; the runner has a single decision loop.
        request = {"model": self.model_name, "mode": "default", **self.options.model_dump()}
        if not (request["catalog_model"] or self.options.base_url or "/" in self.model_name):
            # Revalidate so endpoint options are rejected for presets named only by -m.
            self.options_model.model_validate(
                {**self.options.model_dump(), "catalog_model": self.model_name}
            )
            request["catalog_model"] = self.model_name
        request = await environment.resolve(request, self.extra_env)
        (self.logs_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        context.metadata = {
            "evaluation": "realtime-bench",
            "suite_id": request["task"]["suite_id"],
            "suite_version": request["task"]["suite_version"],
            "model_revision": request["model_revision"],
            "mode": request["mode"],
            "interaction_mode": request["task"]["interaction_mode"],
        }
        try:
            await environment.run_benchmark(request, self.extra_env)
        finally:
            try:
                await environment.download_dir("/logs/agent", self.logs_dir)
            except FileNotFoundError:
                pass  # Setup/identity validation can fail before a runner log exists.
            self.populate_context_post_run(context)

    def populate_context_post_run(self, context):
        populate_context(context, self.logs_dir)


# Environment
"""Apptainer runtime owned by a Harbor trial.

The runner owns GUI instances and their cleanup. Harbor owns the runner
process group, a reserved lane, and trial-local files. Administrative/verifier commands
run in a separate invocation of the installed sandbox image. Agent packages are trusted
local code; this adapter is not an untrusted-code execution service.
"""

import asyncio as asyncio
import fcntl as fcntl
import hashlib as hashlib
import shutil as shutil
import signal as signal
import tempfile as tempfile
import tomllib as tomllib
from pathlib import PurePosixPath as PurePosixPath

from harbor.environments.base import BaseEnvironment as BaseEnvironment
from harbor.environments.base import ExecResult as ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities as EnvironmentCapabilities
from harbor.environments.capabilities import (
    EnvironmentResourceCapabilities as EnvironmentResourceCapabilities,
)


class ApptainerEnvironment(BaseEnvironment):
    def __init__(self, *args, benchmark_root=None, runner_python=None, lane=1, **kwargs):
        self.root = Path(benchmark_root or os.environ.get("RTBENCH_ROOT", ".")).resolve()
        self.python = str(runner_python or os.environ.get("RTBENCH_PYTHON") or sys.executable)
        self.lane = int(lane)
        if not 0 <= self.lane <= 56:
            raise ValueError("lane must be between 0 and 56")
        self.staging = None
        self.task = None
        self.images = None
        self._lock = None
        self._processes = set()
        self._launched = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def type():
        return "realtime-bench-apptainer"

    @property
    def capabilities(self):
        return EnvironmentCapabilities()

    @classmethod
    def resource_capabilities(cls):
        # CPU/RAM are recorded, not enforced by the local runner.
        return EnvironmentResourceCapabilities()

    def _validate_definition(self):
        config = self.environment_dir.parent / "task.toml"
        metadata = tomllib.loads(config.read_text()).get("metadata", {})
        self.task_id = metadata.get("task_id")
        if self.task_id != self.environment_dir.parent.name:
            raise ValueError(f"Not a Realtime-Bench task: {config} must declare its directory name")
        if not (self.root / "realtime_framework/evaluation/harbor.py").is_file():
            raise ValueError("Run Harbor from the benchmark repository root, or set RTBENCH_ROOT")

    async def start(self, force_build=False):
        if not shutil.which("apptainer"):
            raise RuntimeError("Install Apptainer, then run `rtbench setup`")
        # Reserve the lane across Harbor jobs/checkouts on this host. Each sandbox also holds
        # its own lane-local lock, which covers ordinary CLI evaluations.
        locks = Path(tempfile.gettempdir()) / f"rtbench-lanes-{os.getuid()}"
        locks.mkdir(mode=0o700, exist_ok=True)
        self._lock = (locks / f"lane-{self.lane}.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            self._lock = None
            raise RuntimeError("Lane is busy; use -n 1 or assign a different --ek lane") from None
        try:
            listing = await self._run(["apptainer", "instance", "list", "--json"], timeout=30)
            if listing.return_code:
                raise RuntimeError("Cannot inspect existing Apptainer instances")
            active = {entry["instance"] for entry in json.loads(listing.stdout)["instances"]}
            reserved = {f"streaming-agent-sandbox-L{self.lane}", f"rt-realtime-suite-L{self.lane}"}
            if active & reserved:
                raise RuntimeError("Lane has an existing GUI instance; choose another lane")
        except BaseException:
            self._lock.close()
            self._lock = None
            raise
        self.staging = Path(tempfile.mkdtemp(prefix="rtbench-trial-"))
        for directory in ("workspace", "tests", "logs/agent", "logs/verifier", "logs/artifacts"):
            (self.staging / directory).mkdir(parents=True, exist_ok=True)
        await self._upload_environment_dir_after_start()

    def _path(self, path):
        if self.staging is None:
            raise RuntimeError("Environment has not started")
        relative = PurePosixPath(str(path)).relative_to("/")
        if ".." in relative.parts:
            raise ValueError("Path escapes trial directory")
        result = self.staging.joinpath(*relative.parts)
        if not result.resolve().is_relative_to(self.staging):
            raise ValueError("Symlink escapes trial directory")
        return result

    async def _terminate(self, process):
        for sig, seconds in ((signal.SIGINT, 45), (signal.SIGTERM, 10), (signal.SIGKILL, 2)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            # The parent can exit before descendants finish cleanup. Keep the lane
            # reserved until the entire process group has stopped.
            for _ in range(seconds * 10):
                await asyncio.sleep(0.1)
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
            else:
                continue
            break
        await process.wait()

    async def _run(self, command, *, cwd=None, env=None, timeout=None, log=None):
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            start_new_session=True,
            stdout=log if log else asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT if log else asyncio.subprocess.PIPE,
        )
        self._processes.add(process)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
            return ExecResult(
                return_code=process.returncode,
                stdout=stdout.decode(errors="replace") if stdout else "",
                stderr=stderr.decode(errors="replace") if stderr else "",
            )
        except BaseException:
            await asyncio.shield(self._terminate(process))
            raise
        finally:
            self._processes.discard(process)

    def _runner_command(self, request):
        """Resolve and run a Harbor trial in a fresh interpreter."""
        command = [
            self.python,
            "-m",
            "realtime_framework.evaluation._trial",
            "--suite",
            self.task["suite_id"],
            "--task",
            self.task["task_id"],
            "--lane",
            str(self.lane),
            "--fps",
            str(self.task["fps"]),
            "--model-input-fps",
            str(self.task["model_input_fps"]),
            "--interaction-mode",
            self.task["interaction_mode"],
            "--duration-minutes",
            str(self.task["duration_minutes"]),
            "--seed",
            str(self.task["seed"]),
            "--clock", request.get("clock", "wall"),
            "--tokens-per-frame", str(request.get("tokens_per_frame", "unlimited")),
        ]
        if request.get("catalog_model"):
            command += ["--catalog-model", request["catalog_model"]]
        else:
            command += ["--model", request["model"]]
            for field, flag in (
                ("base_url", "--api-base"),
                ("api_key_env", "--api-key-env"),
                ("model_revision", "--model-revision"),
            ):
                if request.get(field):
                    command += [flag, request[field]]
            if not request["stream"]:
                command.append("--no-stream")
        if request.get("reasoning_effort"):
            command += ["--reasoning-effort", request["reasoning_effort"]]
        return command

    def _runner_environment(self, extra_env):
        return {
            **os.environ,
            **(self._merge_env(extra_env) or {}),
            "PYTHONPATH": str(self.root),
            "SANDBOX_IMAGE": str(self.images["sandbox"]),
            "RT_SUITE_IMAGE": str(self.images["realtime"]),
        }

    async def resolve(self, request, extra_env=None):
        """Fix this trial's task identity from the current checkout, then resolve the model."""
        metadata = await asyncio.to_thread(runtime, self.root, self.python, request["suite"])
        suite_tasks = {task["task_id"]: task["category"] for task in metadata["tasks"]}
        if self.task_id not in suite_tasks:
            raise ValueError(f"{self.task_id} is not in suite {request['suite']}; choose --suite")
        self.images = {kind: Path(path) for kind, path in metadata["images"].items()}
        for image in self.images.values():
            if not image.is_file():
                raise FileNotFoundError(f"Missing benchmark image {image}; run `rtbench setup`")
        self.task = {
            "suite_id": metadata["suite_id"],
            "suite_version": metadata["suite_version"],
            "suite_hash": metadata["suite_hash"],
            "harness_version": metadata["harness_version"],
            "harness_hash": metadata["harness_hash"],
            "task_id": self.task_id,
            "category": suite_tasks[self.task_id],
            "fps": request["fps"],
            "model_input_fps": request["model_input_fps"],
            "interaction_mode": request["interaction_mode"],
            "duration_minutes": request["duration_minutes"],
            "seed": request["seed"],
        }
        request = {**request, "task": self.task, "suite_tasks": suite_tasks}
        resolved = await self._run(
            [*self._runner_command(request), "--dry-run"],
            cwd=self.root,
            env=self._runner_environment(extra_env),
            timeout=30,
        )
        if resolved.return_code:
            raise ValueError(f"Cannot resolve model configuration: {resolved.stderr.strip()}")
        launch = json.loads(resolved.stdout)
        from realtime_framework.evaluation.config import configuration_sha256

        # Marks the results of an external driver as unofficial (evaluation/submission.py).
        external = {"external_driver": launch["external_driver"]} if "external_driver" in launch else {}
        return {
            **request,
            "model": launch["model_config"]["model_name"],
            "model_revision": launch.get("model_revision"),
            # Compare with the actual report so edits between resolution and execution
            # cannot silently evaluate a different configuration.
            "configuration_sha256": configuration_sha256(launch),
            **external,
        }

    async def run_benchmark(self, request, extra_env=None):
        if self.task is None or request["task"] != self.task:
            raise ValueError("Resolve the request in this environment before running it")
        # The verifier compares the report with the identity fixed by this environment.
        (self._path("/tests") / "task.json").write_text(json.dumps(self.task, indent=2) + "\n")
        (self._path("/tests") / "verify.py").write_text(verifier_source())
        workspace = self._path("/workspace")
        if (workspace / "request.json").exists():
            raise ValueError("A Harbor trial may run exactly one episode")
        (workspace / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        command = [
            *self._runner_command(request),
            "--report-output",
            str(workspace / "report.json"),
            "--run-root-output",
            str(workspace / "run-root.json"),
        ]
        environment = self._runner_environment(extra_env)
        self._launched = True
        try:
            with self._path("/logs/agent/runner.log").open("wb") as log:
                result = await self._run(command, cwd=self.root, env=environment, log=log)
        finally:
            # Also retain failed/timeout runs before Harbor tears down the environment.
            for name in ("request.json", "report.json"):
                if (workspace / name).is_file():
                    shutil.copyfile(workspace / name, self._path("/logs/artifacts/" + name))
            self._collect_episode_artifacts(request)
        if result.return_code:
            # The verifier still runs and scores the failed episode as an error (reward 0).
            self.logger.warning(
                "Benchmark runner exited %s; see runner.log", result.return_code
            )

    def _collect_episode_artifacts(self, request):
        locator = self._path("/workspace/run-root.json")
        if not locator.is_file():
            return  # Launcher failed before allocating an episode directory.
        try:
            root = Path(json.loads(locator.read_text())["run_root"]).resolve()
            candidates = list(root.glob("*/" + self.task["task_id"]))
            if len(candidates) > 1:
                raise ValueError("More than one episode in this Harbor trial")
            if candidates:
                if not candidates[0].resolve().is_relative_to(root):
                    raise ValueError("Episode directory escapes the allocated run")
                export_episode(candidates[0], self._path("/logs/agent"), request)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Never hide the original execution failure behind an export failure.
            self._path("/logs/agent/artifact-export-error.json").write_text(
                json.dumps({"error": str(exc)}) + "\n"
            )
            self.logger.warning("Could not export benchmark artifacts: %s", exc)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        resolved_user = self._resolve_user(user)
        if resolved_user not in (None, "root", os.getuid(), str(os.getuid())):
            raise ValueError("Apptainer commands run as the invoking host user")
        # "root" is Harbor's administrative-command convention; it never changes UID.
        environment = os.environ.copy()
        for key, value in (self._merge_env(env) or {}).items():
            environment["APPTAINERENV_" + key] = value
        command_line = ["apptainer", "exec", "--cleanenv", "--containall", "--no-home"]
        for folder in ("workspace", "tests", "logs"):
            command_line += ["--bind", f"{self._path('/' + folder)}:/{folder}"]
        command_line += [
            "--pwd",
            cwd or "/workspace",
            str(self.images["sandbox"]),
            "bash",
            "-c",
            command,
        ]
        return await self._run(command_line, env=environment, timeout=timeout_sec)

    async def upload_file(self, source_path, target_path):
        target = self._path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, self._path(target_dir), dirs_exist_ok=True)

    async def download_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(source_path), target_path)

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(self._path(source_dir), target_dir, dirs_exist_ok=True)

    async def stop(self, delete=True):
        try:
            for process in list(self._processes):
                await self._terminate(process)
            if self._launched:
                result = await self._run(
                    [
                        self.python,
                        "-m",
                        "realtime_framework.sandbox.apptainer",
                        "stop",
                        f"streaming-agent-sandbox-L{self.lane}",
                        f"rt-realtime-suite-L{self.lane}",
                    ],
                    cwd=self.root,
                    env={**os.environ, "PYTHONPATH": str(self.root)},
                    timeout=75,
                )
                if result.return_code:
                    raise RuntimeError("Could not clean up the trial GUI instances")
                self._launched = False
            if delete and self.staging:
                shutil.rmtree(self.staging)
                self.staging = None
        finally:
            if self._lock:
                fcntl.flock(self._lock, fcntl.LOCK_UN)
                self._lock.close()
                self._lock = None


# Report
"""Aggregate one recorded Harbor evaluation; incomplete suites have no headline score."""


import yaml

ROOT = Path(__file__).resolve().parents[2]


def official_suite(identity, suite_tasks):
    """Accept only a released private suite whose content and membership match this checkout,
    run on a released harness."""
    from realtime_framework.evaluation.config import HARNESS_UNRELEASED
    from realtime_tasks.registry import suite_path

    selector = f"{identity['suite_id']}@{identity.get('suite_version')}"
    if identity["suite_id"] != "private":
        raise ValueError(f"Official evaluation requires the private suite, not {selector}")
    try:
        manifest = yaml.safe_load(suite_path(ROOT, selector).read_text())
    except ValueError as exc:
        raise ValueError(f"Official evaluation requires a released suite: {exc}") from exc
    released = {
        task: group["public_category"]
        for group in manifest["categories"]
        for task in group["task_ids"]
    }
    if manifest.get("suite_hash") != identity["suite_hash"] or suite_tasks != released:
        raise ValueError(f"Official evaluation requires the entire released {selector} suite")
    harness = identity["harness_version"]
    if harness.endswith(HARNESS_UNRELEASED):
        raise ValueError(
            f"Official evaluation requires a released harness, not harness@{harness}, whose "
            "framework sources or runtime dependencies differ from the release pyproject.toml "
            "locks"
        )


def trial_error_reason(directory):
    """The verifier's recorded error reason, else Harbor's exception type for this trial."""
    try:
        verdict = json.loads((directory / "verifier/evaluation.json").read_text())
        if verdict.get("error_reason"):
            return verdict["error_reason"]
    except (OSError, ValueError, AttributeError):
        pass
    try:
        info = json.loads((directory / "result.json").read_text()).get("exception_info") or {}
        if info.get("exception_type"):
            return f"harbor_{info['exception_type']}"
    except (OSError, ValueError, AttributeError):
        pass
    return "unverified_trial"


def outcome_counts(rows):
    counts = {name: 0 for name in ("pass", "fail", "timeout", "error")}
    for row in rows:
        counts[row["outcome"]] += 1
    return counts


def aggregate(job, *, official=False):
    """Score a job against the suite membership and identity recorded by its trials."""
    requests, diagnostics = {}, []
    for directory in sorted(Path(job).iterdir()):
        if not directory.is_dir():
            continue
        request_path = directory / "agent/request.json"
        if request_path.is_file():
            requests[directory] = json.loads(request_path.read_text())
        elif (directory / "result.json").is_file():
            # Environment/setup errors occur before the request exists.
            diagnostics.append("Trial failed before integration request was recorded")
    if not requests:
        raise ValueError("The job has no recorded trials")
    shared_fields = (
        "suite_id", "suite_version", "suite_hash", "harness_version", "harness_hash", "fps",
        "model_input_fps", "interaction_mode", "duration_minutes", "seed",
    )
    identities = {
        json.dumps(
            # Jobs recorded before suite versions have no suite_version.
            [{k: r["task"][k] for k in shared_fields if k in r["task"]}, r["suite_tasks"]],
            sort_keys=True,
        )
        for r in requests.values()
    }
    if len(identities) != 1:
        raise ValueError("The job mixes benchmark versions, suites or runtime settings")
    shared, suite_tasks = json.loads(next(iter(identities)))
    tasks = {
        task_id: {**shared, "task_id": task_id, "category": category}
        for task_id, category in suite_tasks.items()
    }
    suite_id = shared["suite_id"]
    if official:
        official_suite(shared, suite_tasks)
    rows = {
        task: {
            "task_id": task,
            "category": value["category"],
            "status": "missing",
            "outcome": "error",
            "error_reason": "missing_trial",
        }
        for task, value in tasks.items()
    }
    systems, seen, configuration_hashes = set(), set(), set()
    runtime_hashes = set()
    for directory, request in requests.items():
        task_id = request["task"]["task_id"]
        if task_id not in tasks or task_id in seen:
            raise ValueError("Unexpected or repeated task; retries require a separate job")
        seen.add(task_id)
        system = {k: v for k, v in request.items() if k not in ("task", "suite_tasks")}
        systems.add(json.dumps(system, sort_keys=True))
        row = rows[task_id]
        try:
            report = json.loads((directory / "artifacts/logs/artifacts/report.json").read_text())
            reward = verify(report, tasks[task_id], request)
            recorded = json.loads((directory / "verifier/reward.json").read_text())
            result = json.loads((directory / "result.json").read_text())
            if recorded != reward or result.get("exception_info") is not None:
                raise ValueError("Verifier reward or trial completion mismatch")
            configuration_hashes.add(report["integration"]["configuration_sha256"])
            runtime_hashes.add(report["groups"][0]["runtime_config_hash"])
            row.update(
                status="scored",
                solved=bool(reward["reward"]),
                outcome=report["groups"][0]["tasks"][0]["outcome"],
                error_reason=None,
            )
        except (OSError, ValueError, KeyError, TypeError):
            # An error is scored as a failure; its reason stays visible.
            row.update(
                status="error",
                solved=False,
                outcome="error",
                error_reason=trial_error_reason(directory),
            )
    if len(systems) > 1 or len(configuration_hashes) > 1:
        raise ValueError("Cannot aggregate different model revisions, modes or configurations")
    # Every task needs a trial; an errored trial counts as an unsolved task.
    complete = len(seen) == len(tasks) and not diagnostics
    groups = []
    for category in sorted({t["category"] for t in tasks.values()}):
        group = [r for r in rows.values() if r["category"] == category]
        ran = [r for r in group if r["status"] != "missing"]
        groups.append(
            {
                "category": category,
                "expected": len(group),
                "scored": len(ran),
                "outcomes": outcome_counts(group),
                "passed": sum(r["solved"] for r in ran),
                "success_rate": sum(r["solved"] for r in ran) / len(group)
                if len(ran) == len(group)
                else None,
            }
        )
    system = json.loads(next(iter(systems))) if systems else {}
    identity = shared
    return {
        "schema": "realtime-bench-report-v1",
        "evaluation": "maintainer_private" if official else "self_reported",
        "suite_id": suite_id,
        "suite_version": identity.get("suite_version"),
        "suite_hash": identity["suite_hash"],
        "harness_version": identity.get("harness_version"),
        "harness_hash": identity["harness_hash"],
        "fps": identity["fps"],
        # Jobs recorded before the frame rates were split fed the model at the environment rate.
        "model_input_fps": identity.get("model_input_fps", identity["fps"]),
        # Jobs recorded before the interaction mode was a run setting ran turn-based.
        "interaction_mode": identity.get("interaction_mode", "turn_based"),
        "duration_minutes": identity["duration_minutes"],
        "clock": system.get("clock", "wall"),
        **(
            {}
            if system.get("clock", "wall") == "wall"
            else {"tokens_per_frame": system.get("tokens_per_frame", "unlimited")}
        ),
        "seed": identity["seed"],
        "model": system.get("model"),
        "model_revision": system.get("model_revision"),
        "mode": system.get("mode"),
        **({"external_driver": system["external_driver"]} if "external_driver" in system else {}),
        "integration_hash": hashlib.sha256(next(iter(systems)).encode()).hexdigest()
        if systems
        else None,
        "system_configuration_sha256": next(iter(configuration_hashes), None),
        "runtime_config_hashes": sorted(runtime_hashes),
        "complete": complete,
        "expected": len(tasks),
        "scored": sum(r["status"] != "missing" for r in rows.values()),
        "outcomes": outcome_counts(rows.values()),
        "success_rate": sum(r.get("solved", False) for r in rows.values()) / len(tasks)
        if complete
        else None,
        "categories": groups,
        "diagnostics": diagnostics,
        # Private task identities and raw artifacts stay with the maintainers.
        **({} if official else {"tasks": list(rows.values())}),
    }


def harbor_report_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official", action="store_true", help="Maintainer-only private report")
    args = parser.parse_args(argv)
    report = aggregate(args.job, official=args.official)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["complete"]:
        print(
            f"Incomplete evaluation: {report['scored']}/{report['expected']} tasks scored, "
            f"{len(report['diagnostics'])} diagnostics; no headline score. Wrote {args.output}",
            file=sys.stderr,
        )
    return 0 if report["complete"] else 1
