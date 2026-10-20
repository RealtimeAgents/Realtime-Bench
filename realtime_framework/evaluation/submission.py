"""Check finished public runs and upload them to the Realtime-Bench submission server.

``rtbench upload`` replaces Harbor's Hub upload: results go to the benchmark's own server
(``RTBENCH_SUBMIT_URL``), authenticated with a per-submitter token (``RTBENCH_SUBMIT_TOKEN``).
The archive is uploaded in chunks; rerunning the same command resumes an interrupted upload.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from realtime_framework.evaluation.harbor import aggregate
from realtime_tasks.registry import UNRELEASED

DEFAULT_URL = "https://realtimeagents.org/api/submissions"
RUNS = 3
# The leaderboard setting; any other value is a different benchmark condition.
REQUIRED = {"suite_id": "public", "clock": "wall", "fps": 1, "duration_minutes": 5.0, "seed": 42}
# Fields every run of one submission must share.
SAME = (
    "suite_version", "suite_hash", "harness_version", "harness_hash", "model", "model_revision",
    "system_configuration_sha256", "integration_hash",
)
RETRIES = 5


class SubmissionError(RuntimeError):
    pass


def check_runs(jobs: list[Path]) -> list[dict]:
    """Return one report per job, or explain why the jobs are not a leaderboard submission."""
    if len(jobs) != RUNS or len({job.resolve() for job in jobs}) != RUNS:
        raise SubmissionError(f"Give {RUNS} different job directories, one per run")
    reports = []
    for job in jobs:
        try:
            report = aggregate(job)
        except (OSError, ValueError, KeyError) as exc:
            raise SubmissionError(f"{job}: {exc}") from exc
        if not report["complete"]:
            raise SubmissionError(f"{job}: {report['scored']}/{report['expected']} tasks have a trial")
        for field, value in REQUIRED.items():
            if report.get(field) != value:
                raise SubmissionError(f"{job}: {field} is {report.get(field)!r}, expected {value!r}")
        for field in ("suite_version", "harness_version"):
            if str(report.get(field) or UNRELEASED).endswith(UNRELEASED):
                raise SubmissionError(f"{job}: {field} {report.get(field)!r} is not a release")
        reports.append(report)
    for field in SAME:
        if len({json.dumps(report.get(field)) for report in reports}) > 1:
            raise SubmissionError(f"The runs differ in {field}; submit one configuration")
    return reports


def build_archive(jobs: list[Path], reports: list[dict], output: Path) -> dict:
    """Write ``submission.json`` and every job directory into one gzip tarball."""
    manifest = {
        "schema": "realtime-bench-submission-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [{"job": job.name, "report": report} for job, report in zip(jobs, reports)],
    }
    data = json.dumps(manifest, indent=2).encode()
    with tarfile.open(output, "w:gz", compresslevel=1) as archive:
        info = tarfile.TarInfo("submission.json")
        info.size, info.mtime = len(data), int(time.time())
        archive.addfile(info, io.BytesIO(data))
        for index, job in enumerate(jobs, 1):
            archive.add(
                job, arcname=f"jobs/run{index}-{job.name}",
                filter=lambda member: None if "/.harbor-upload" in member.name else member,
            )
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Client:
    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    def request(self, method: str, path: str = "", body: bytes | None = None, **headers) -> dict:
        request = urllib.request.Request(
            self.url + path, data=body, method=method,
            headers={"Authorization": f"Bearer {self.token}", **headers},
        )
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return json.loads(response.read() or b"{}")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                if exc.code < 500 or attempt == RETRIES - 1:
                    raise SubmissionError(f"{method} {path or '/'}: HTTP {exc.code} {detail}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == RETRIES - 1:
                    raise SubmissionError(f"{method} {path or '/'}: {exc}") from exc
            time.sleep(2**attempt)
        raise AssertionError("unreachable")


def upload(archive: Path, manifest: dict, client: Client, log=print) -> dict:
    size, sha256 = archive.stat().st_size, _sha256(archive)
    first = manifest["runs"][0]["report"]
    created = client.request(
        "POST", body=json.dumps({
            "size": size, "sha256": sha256, "model": first.get("model"),
            "suite": f"{first['suite_id']}@{first['suite_version']}",
            "harness": first.get("harness_version"),
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        }).encode(), **{"Content-Type": "application/json"},
    )
    submission, chunk = created["id"], created["chunk_size"]
    received = created["received"]
    if received:
        log(f"Resuming {submission} at {received}/{size} bytes")
    with archive.open("rb") as stream:
        while received < size:
            stream.seek(received)
            block = stream.read(chunk)
            try:
                received = client.request(
                    "PUT", f"/{submission}?offset={received}", block,
                    **{"Content-Type": "application/octet-stream"},
                )["received"]
            except SubmissionError:
                # A lost response may have stored the chunk; continue from the server's offset.
                received = client.request("GET", f"/{submission}")["received"]
            log(f"  {received / size:6.1%}  {received}/{size} bytes")
    return client.request("POST", f"/{submission}/complete")


def main(jobs: list[Path], *, url: str | None = None, token: str | None = None,
         keep_archive: bool = False) -> dict:
    from realtime_framework.providers.registry import load_project_secrets

    load_project_secrets()
    url = url or os.environ.get("RTBENCH_SUBMIT_URL") or DEFAULT_URL
    token = token or os.environ.get("RTBENCH_SUBMIT_TOKEN")
    reports = check_runs(jobs)
    print(f"Checked {RUNS} runs of {reports[0]['model']}: "
          + ", ".join(f"{r['outcomes'].get('pass', 0)}/{r['expected']} passed" for r in reports))
    if not token:
        raise SubmissionError(
            "The runs are valid. Set RTBENCH_SUBMIT_TOKEN in .env to upload; "
            "request a token from realtime-agents@utlists.utexas.edu"
        )
    folder = Path("temp/submissions")
    folder.mkdir(parents=True, exist_ok=True)
    # The archive name is stable for the same jobs, so a rerun resumes the same upload.
    key = hashlib.sha256("\n".join(str(job.resolve()) for job in jobs).encode()).hexdigest()[:16]
    archive = folder / f"{key}.tar.gz"
    manifest_path = archive.with_suffix(".json")
    if archive.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        print("Packing results...")
        manifest = build_archive(jobs, reports, archive)
        manifest_path.write_text(json.dumps(manifest))
    result = upload(archive, manifest, Client(url, token))
    if not keep_archive:
        archive.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    return result
