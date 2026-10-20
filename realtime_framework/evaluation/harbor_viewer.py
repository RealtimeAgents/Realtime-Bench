"""Harbor result presentation: ATIF export of episode evidence and the benchmark viewer.

Nothing here affects rewards; scoring is ``harbor.verify`` over the runner's report.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from harbor.models.trajectories.trajectory import Trajectory
from harbor.models.trial.result import TimingInfo
from harbor.viewer.scanner import JobScanner

from realtime_framework.evaluation.harbor_results import job_label, job_results
from realtime_framework.evaluation.reporting import active_interval, browser_metadata, read_json

__version__ = "0.4.0"
VIEWER_SCRIPT = Path(__file__).parent / "viewer/rtbench-viewer.js"
TOKEN_FIELDS = (
    "provider_input_tokens",
    "provider_output_tokens",
    "provider_cached_input_tokens",
)


def _json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def _rows(path):
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed writer can leave the final record incomplete. Keep its raw file.
            if index != len(lines) - 1:
                raise
    return result


def _timestamp(seconds):
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _total(summary, field, values):
    count = ((summary.get("token_accounting") or {}).get("counts") or {}).get(field)
    if count is not None:
        return count.get("total") if count.get("complete") else None
    return sum(values) if values and all(value is not None for value in values) else None


def export_episode(run_dir, logs_dir, request):
    """Copy evidence and convert it without changing source records or scores.

    Screenshots are matched by digest, not decision number: capture and inference can
    run at different rates. Stored captures precede the adapter's resize/mark overlay.
    ATIF describes harness-visible decisions, which need not map one-to-one to model
    invocations. Missing metrics remain unknown.
    """
    run_dir, logs_dir = Path(run_dir), Path(logs_dir)

    def retain(source, destination):
        if destination.exists():
            return
        shutil.copy2(source, destination)

    summary = _json(run_dir / "run_summary.json")
    if summary and summary.get("task_id") != request["task"]["task_id"]:
        raise ValueError("Artifact task identity differs from the Harbor trial")
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw = logs_dir / "benchmark"
    raw.mkdir(exist_ok=True)
    names = (
        "run_summary.json",
        "run_config.json",
        "evaluation.json",
        "trajectory.jsonl",
        "action_lifecycle.jsonl",
        "episode_trace.jsonl",
        "tool_calls.jsonl",
        "runner.out",
        "deferred-provider-usage.jsonl",
        "provider-usage.jsonl",
        "active-loop.json",
        "recording-timing.json",
        "recording-timeline.json",
    )
    for name in names:
        source = run_dir / name
        if source.is_file() and not source.is_symlink():
            retain(source, raw / name)
    recording = run_dir / "video.mp4"
    # Export the captured original; the viewer draws annotations in the browser.
    recording_metadata = {
        "schema": "browser-overlay-v1",
        "state": "unavailable",
        "reason": "No original video was recorded or preserved",
        "video_source": "original",
        "baked_annotations": False,
    }
    if recording.is_file():
        recording_metadata = browser_metadata(run_dir)
        recording_metadata["video_source"] = "original"
        stat = recording.stat()
        recording_metadata["source_signature"] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        destination = logs_dir / "recording.mp4"
        if destination.exists() and not destination.samefile(recording):
            destination.unlink()
        retain(recording, destination)
    (logs_dir / "recording-metadata.json").write_text(
        json.dumps(recording_metadata, indent=2) + "\n"
    )

    rows = [
        row
        for row in _rows(raw / "trajectory.jsonl")
        if row.get("type") == "decision" or row.get("parsed_action") is not None
    ]
    lifecycle = {row["decision"]: row for row in _rows(raw / "action_lifecycle.jsonl")}
    events = _rows(raw / "episode_trace.jsonl")
    captures = {
        event["payload"]["observation_id"]: event["payload"]
        for event in events
        if event.get("event") == "ObservationCaptured"
    }
    inspections = {row.get("observation_id"): row for row in _rows(raw / "tool_calls.jsonl")}
    referenced = set()
    for row in rows:
        life = lifecycle.get(row.get("decision", row.get("tick")), {})
        oid = row.get("observation_id") or life.get("observation_id")
        referenced.add(
            (captures.get(oid) or {}).get("screenshot_sha256")
            or (inspections.get(oid) or {}).get("screenshot_sha256")
        )
    images = {}
    for source in sorted((run_dir / "frames").glob("*.png")):
        if source.is_symlink():
            continue
        image_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if image_digest not in referenced:
            continue
        destination = logs_dir / "images" / source.name
        destination.parent.mkdir(exist_ok=True)
        retain(source, destination)
        images[image_digest] = destination.relative_to(logs_dir).as_posix()

    model = summary.get("backend_model") or request["model"]
    steps = [
        {
            "step_id": 1,
            "source": "user",
            "message": summary.get("task_description") or request["task"]["task_id"],
        }
    ]
    tokens_rows = []
    for row in rows:
        decision = row.get("decision", row.get("tick"))
        life = lifecycle.get(decision, {})
        observation_id = row.get("observation_id") or life.get("observation_id")
        capture = captures.get(observation_id, {})
        inspection = inspections.get(observation_id, {})
        parts = [
            {
                "type": "text",
                "text": (
                    f"Observation {observation_id}, tick {row.get('observation_tick')}.\n"
                    "Screenshot is the captured screen before adapter resizing and mark overlay.\n"
                    + inspection.get("delivered_text", "")
                ),
            }
        ]
        digest = capture.get("screenshot_sha256") or inspection.get("screenshot_sha256")
        if digest in images:
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "media_type": "image/png",
                        "path": images[digest],
                    },
                }
            )
        steps.append(
            {
                "step_id": len(steps) + 1,
                "source": "user",
                "message": parts,
                "timestamp": _timestamp(capture.get("timestamp")),
                "extra": {"observation_id": observation_id, "screenshot_sha256": digest},
            }
        )

        accounting = life.get("token_accounting") or row.get("usage") or {}
        tokens = {field: accounting.get(field) for field in TOKEN_FIELDS}
        tokens_rows.append(tokens)
        # Usage stays out of Harbor's step metrics: wall-clock scores do not depend on it,
        # and Harbor would present it, or a zero total derived from it, as token usage.
        measurements = {
            "provider_tokens": tokens,
            **{
                key: row.get(key)
                for key in (
                    "decision_wall_latency_s",
                    "provider_call_wall_latency_s",
                    "provider_ttft_s",
                    "provider_reasoning_output_tokens",
                )
            },
        }
        action = row.get("parsed_action")
        call_id = f"gui-{decision}"
        tool_calls = (
            [
                {
                    "tool_call_id": call_id,
                    "function_name": "gui_action",
                    "arguments": action,
                    "extra": {
                        "serialization": "harness_action",
                        "action_protocol": row.get("action_protocol"),
                    },
                }
            ]
            if action
            else None
        )
        feedback = {
            key: life.get(key)
            for key in (
                "status",
                "execution_result",
                "coordinate_trace",
                "terminal_after_action",
                "action_clock_position",
                "action_resolved_unix_s",
            )
        }
        steps.append(
            {
                "step_id": len(steps) + 1,
                "source": "agent",
                "model_name": model,
                "reasoning_effort": summary.get("reasoning_effort"),
                "timestamp": _timestamp(life.get("action_resolved_unix_s")),
                "message": row.get("model_output") or "",
                "reasoning_content": row.get("thinking") or None,
                "tool_calls": tool_calls,
                "observation": {
                    "results": [
                        {
                            "source_call_id": call_id if action else None,
                            "content": json.dumps(feedback, ensure_ascii=False),
                        }
                    ]
                },
                "extra": {
                    **measurements,
                    "decision": decision,
                    "observation_id": observation_id,
                    "logged_harness_context": row.get("input_prompt"),
                    "runtime": row.get("realtime_runtime"),
                },
            }
        )

    provider_tokens = {
        field: _total(summary, field, [row[field] for row in tokens_rows])
        for field in TOKEN_FIELDS
    }
    metadata = {
        "evaluation": "realtime-bench",
        "suite_id": request["task"]["suite_id"],
        "suite_version": request["task"].get("suite_version"),
        "task_id": request["task"]["task_id"],
        "mode": request["mode"],
        "model_revision": request.get("model_revision"),
        "reasoning_effort": summary.get("reasoning_effort"),
        "configuration_sha256": request.get("configuration_sha256"),
        "run_status": summary.get("run_status"),
        "outcome": summary.get("outcome"),
        "error_reason": summary.get("error_reason"),
        "solved": summary.get("solved"),
        "decisions": summary.get("decisions"),
        "actions_executed": summary.get("actions_executed"),
        "actions_failed": summary.get("actions_failed"),
        "wall_time_s": summary.get("wall_time_s"),
        "model_timing": summary.get("model_timing"),
        "total_run_time_s": summary.get("total_run_time_s"),
        "loop_time_s": summary.get("loop_time_s"),
        "active_loop": _json(run_dir / "active-loop.json"),
        "recording": {
            key: recording_metadata.get(key) for key in ("schema", "state", "output_duration_s")
        },
        "provider_tokens": provider_tokens,
        "token_accounting_scope": (summary.get("token_accounting") or {}).get("scope"),
        "cost_usd": summary.get("cost_usd"),
        "cost_status": summary.get("cost_status"),
        "source_run": str(run_dir.resolve()),
    }
    trajectory = Trajectory.model_validate(
        {
            "schema_version": "ATIF-v1.8",
            "session_id": run_dir.parent.parent.name + "/" + run_dir.name,
            "agent": {"name": "Realtime-Bench Agent", "version": __version__, "model_name": model},
            "steps": steps,
            "final_metrics": {"total_steps": len(steps)},
            "extra": metadata,
            "notes": (
                "Converted from benchmark evidence. Agent steps are harness-visible decisions, "
                "not necessarily one internal model call. GUI tools represent submitted actions; "
                "execution feedback records whether they were applied. Messages are the harness "
                "recorded outputs. Screenshots are "
                "raw captures before adapter resizing/mark overlay. Full logged harness contexts "
                "(which can include the current response) are in step.extra. Raw evidence is in "
                "benchmark/; recording.mp4 preserves the source recording. The benchmark recording "
                "player limits playback to the active loop and draws annotations in the browser. "
                "recording-metadata.json describes playback alignment. "
                "Thinking labels indicate an outstanding model call, not verified internal reasoning. "
                "Provider token counts and latencies are in each step's extra; missing values are unknown."
            ),
        }
    )
    (logs_dir / "trajectory.json").write_text(
        json.dumps(trajectory.to_json_dict(), ensure_ascii=False, indent=2) + "\n"
    )
    context = {"metadata": metadata}
    (logs_dir / "agent-context.json").write_text(json.dumps(context, indent=2) + "\n")
    return context


def populate_context(context, logs_dir):
    values = _json(Path(logs_dir) / "agent-context.json")
    for key, value in values.items():
        if key == "metadata":
            value = {**(context.metadata or {}), **value}
        setattr(context, key, value)
    error = _json(Path(logs_dir) / "artifact-export-error.json")
    if error:
        context.metadata = {**(context.metadata or {}), "artifact_export_error": error}



# Viewer


# Files a run window is derived from; their change invalidates the cached window.
RUN_WINDOW_EVIDENCE = (
    "active-loop.json",
    "run_summary.json",
    "episode_trace.jsonl",
    "action_lifecycle.jsonl",
)


@lru_cache(maxsize=8192)
def _agent_run(evidence: Path, signature: tuple) -> TimingInfo | None:
    try:
        start, end, _ = active_interval(evidence)
    except (OSError, ValueError, KeyError):
        return None  # Setup-only or incomplete episode: no agent run to present.
    return TimingInfo(
        started_at=datetime.fromtimestamp(start, timezone.utc),
        finished_at=datetime.fromtimestamp(end, timezone.utc),
    )


class AgentRunScanner(JobScanner):
    """Present each benchmark trial by its agent run, the scored active loop.

    Harbor times the whole trial, so its durations would include sandbox start-up and
    teardown, which happen inside agent execution. Every Harbor duration derives from a
    trial's start and end, so presenting those as the run window makes trial durations, task
    averages and the timing panel all describe the agent run that the recording shows.
    result.json keeps Harbor's own times; trials without a run window keep them too.
    """

    def get_trial_result(self, job_name, trial_name):
        result = super().get_trial_result(job_name, trial_name)
        evidence = self.jobs_dir / job_name / trial_name / "agent/benchmark"
        if result is None or not evidence.is_dir():
            return result
        signature = tuple(
            (path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else None
            for path in (evidence / name for name in RUN_WINDOW_EVIDENCE)
        )
        run = _agent_run(evidence, signature)
        if run is None:
            return result
        return result.model_copy(
            update={
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "environment_setup": None,
                "agent_setup": None,
                "agent_execution": run,
                "verifier": None,
            }
        )


def create_app(folder: Path, *, static_dir: Path | None = None):
    from fastapi import HTTPException as HTTPException
    from fastapi.responses import FileResponse as FileResponse
    from fastapi.responses import HTMLResponse as HTMLResponse
    from fastapi.staticfiles import StaticFiles as StaticFiles
    import harbor.viewer.server as harbor_server

    folder = Path(folder).resolve()
    static = static_dir or Path(harbor_server.__file__).parent / "static"
    # Harbor's server constructs its scanner by name; give it the benchmark's presentation.
    harbor_server.JobScanner = AgentRunScanner
    try:
        app = harbor_server.create_app(folder, static_dir=None)
    finally:
        harbor_server.JobScanner = JobScanner

    @app.get("/api/rtbench/jobs")
    def result_labels():
        return {
            path.name: job_label(path)
            for path in sorted(folder.iterdir())
            if path.is_dir() and (path / "config.json").is_file()
        } if folder.is_dir() else {}

    @app.get("/api/rtbench/jobs/{job_name}")
    def results(job_name: str):
        if job_name in {"", ".", ".."} or "/" in job_name or "\\" in job_name:
            raise HTTPException(400, "Invalid job")
        job = (folder / job_name).resolve()
        if not job.is_relative_to(folder) or not job.is_dir():
            raise HTTPException(404, "Job not found")
        return job_results(job)

    @lru_cache(maxsize=128)
    def load_timeline(agent_path, signature):
        agent = Path(agent_path)
        meta = read_json(agent / "recording-metadata.json")
        if meta.get("schema") == "browser-overlay-v1" and (
            meta.get("baked_annotations")
            or (meta.get("annotation_policy") or {}).get("target_boxes_version") == 1
        ):
            return meta
        # A completed active-overlay manifest links an already rendered recording.
        # Do not draw over its embedded captions or crop it again.
        processed = (
            meta
            if meta.get("state") == "completed"
            and str(meta.get("schema", "")).startswith("active-overlay-")
            else None
        )
        return browser_metadata(agent / "benchmark", processed=processed)

    @app.get("/api/jobs/{job_name}/trials/{trial_name}/recording/timeline")
    def timeline(job_name: str, trial_name: str):
        if any(
            value in {"", ".", ".."} or "/" in value or "\\" in value
            for value in (job_name, trial_name)
        ):
            raise HTTPException(400, "Invalid trial path")
        agent = (folder / job_name / trial_name / "agent").resolve()
        if not agent.is_relative_to(folder) or not agent.is_dir():
            raise HTTPException(404, "Trial not found")
        paths = [agent / "recording-metadata.json", agent / "recording.mp4"]
        paths += [
            agent / "benchmark" / name
            for name in (
                "run_summary.json",
                "episode_trace.jsonl",
                "action_lifecycle.jsonl",
                "trajectory.jsonl",
                "tool_calls.jsonl",
                "runner.out",
                "recording-timing.json",
                "active-loop.json",
            )
        ]
        signature = []
        for path in paths:
            if not path.resolve().is_relative_to(folder):
                raise HTTPException(400, "Recording evidence must remain inside the jobs directory")
            if path.exists():
                stat = path.stat()
                signature.append((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns))
        return load_timeline(str(agent), tuple(signature))

    @app.get("/rtbench-viewer.js")
    def viewer_module():
        return FileResponse(
            VIEWER_SCRIPT, media_type="text/javascript", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/rtbench-results.js")
    def results_module():
        return FileResponse(
            VIEWER_SCRIPT.with_name("rtbench-results.js"), media_type="text/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    for name in ("assets", "fonts"):
        if (static / name).is_dir():
            app.mount("/" + name, StaticFiles(directory=static / name), name=name)

    @app.get("/favicon.ico")
    def favicon():
        return FileResponse(static / "favicon.ico")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "API route not found")
        # Harbor's UI is served unchanged; rtbench-viewer.js adds the benchmark enhancements.
        html = (static / "index.html").read_text()
        return HTMLResponse(
            html.replace(
                "</body>", '<script type="module" src="/rtbench-viewer.js"></script></body>'
            ),
            headers={"Cache-Control": "no-cache"},
        )

    @app.middleware("http")
    async def revalidate_assets(request, call_next):
        # Never reuse a cached bundle patched under Harbor's file names.
        response = await call_next(request)
        if request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    return app
