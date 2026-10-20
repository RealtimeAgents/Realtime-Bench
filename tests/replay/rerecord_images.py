"""Re-record every replay baseline and check that, against a git revision, only image digests
changed. This moves baselines recorded with the SHA-256 of each image's PNG bytes to its pixel
digest (baseline.image_digest), on any tree whose baselines are still in the old form:

    .venv/bin/python -m tests.replay.rerecord_images [--against REV] [--check-only]

Every tests/replay/test_*.py that honours REPLAY_RECORD records, one module at a time;
test_chat_completions.py goes first, since test_external_driver.py compares itself with
chat_completions-nonstream-long.jsonl in either mode (test_external_driver.py:203). The check
expands each file's storage (message, blob and block references) at REV and in the working tree
and walks both together. Every difference must be an image whose old value, the SHA-256 of its
PNG bytes, became a pixel digest, each old SHA-256 must become the same digest everywhere, and
none may be left in any recorded file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from . import baseline, baseline_decisions

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
BASELINES = HERE / "baselines"

PNG_SHA256 = re.compile(r"[0-9a-f]{64}")
DIGEST = re.compile(r"rgba:\d+x\d+:[0-9a-f]{64}")
OLD_DATA_URL = re.compile(r"data:([^;,]+);sha256,([0-9a-f]{64})")  # the old baseline._string
OLD_BLOB = re.compile(r"sha256:([0-9a-f]{64})")  # the old Anthropic sources and Gemini blobs


def record_modules() -> list[Path]:
    modules = sorted(p for p in HERE.glob("test_*.py") if "REPLAY_RECORD" in p.read_text())
    first = HERE / "test_chat_completions.py"
    return [first] + [p for p in modules if p != first]


def record(modules: list[Path]) -> list[str]:
    """The names of the modules whose recording failed."""
    failed = []
    for module in modules:
        print(f"== recording {module.name}", flush=True)
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", str(module)]
        environment = {**os.environ, "REPLAY_RECORD": "1"}
        if subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode:
            failed.append(module.name)
    return failed


def load(text: str) -> list[dict]:
    """A baseline's records with its storage expanded, as the comparing test reads it."""
    lines = [json.loads(line) for line in text.splitlines()]
    kinds = {line["record"] for line in lines}
    if "block" in kinds:  # test_canonical_snapshot.snapshot
        blocks = {line["key"]: line["block"] for line in lines if line["record"] == "block"}
        return [
            {**line, "items": [{**item, "block": blocks[item["block"]]} for item in line["items"]]}
            for line in lines
            if line["record"] == "request"
        ]
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "baseline.jsonl"
        path.write_text(text, encoding="utf-8")
        return (baseline_decisions if "blob" in kinds else baseline).read(path)


def image_change(old, new, key: str | None) -> tuple[str, str] | None:
    """(PNG SHA-256, pixel digest) where an old value held the one and the new value the other."""
    if not isinstance(new, str):
        return None
    if isinstance(old, dict):  # a canonical request's image bytes
        if old.keys() == {"byte_length", "sha256"} and DIGEST.fullmatch(new):
            return old["sha256"], new
        return None
    if not isinstance(old, str):
        return None
    if key in baseline.IMAGE_SHA256_FIELDS and PNG_SHA256.fullmatch(old) and DIGEST.fullmatch(new):
        return old, new
    url = OLD_DATA_URL.fullmatch(old)
    if url and new.startswith(f"data:{url[1]};") and DIGEST.fullmatch(new.split(";", 1)[1]):
        return url[2], new.split(";", 1)[1]
    blob = OLD_BLOB.fullmatch(old)
    if key == "data" and blob and DIGEST.fullmatch(new):
        return blob[1], new
    return None


class Diff:
    """The differences of one file; ``digests`` collects, across files, each old PNG SHA-256's
    new digests."""

    def __init__(self, digests: dict[str, set[str]]):
        self.digests = digests
        self.images = 0
        self.problems: list[str] = []

    def walk(self, old, new, path: str, key: str | None = None) -> None:
        if old == new:
            return
        change = image_change(old, new, key)
        if change is not None:
            sha, digest = change
            self.digests[sha].add(digest)
            self.images += 1
        elif isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
            for name in sorted(old):
                self.walk(old[name], new[name], f"{path}.{name}", name)
        elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
            for index, (a, b) in enumerate(zip(old, new)):
                self.walk(a, b, f"{path}[{index}]")
        else:
            self.problems.append(f"{path}: {show(old)} -> {show(new)}")


def show(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 160 else text[:157] + "..."


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def check(against: str) -> bool:
    folder = BASELINES.relative_to(ROOT).as_posix()
    before = {Path(p).name for p in git("ls-tree", "--name-only", against, f"{folder}/").split()}
    after = {p.name for p in BASELINES.glob("*.jsonl")}
    ok = before == after
    for name in sorted(before - after):
        print(f"{name}: missing from the working tree")
    for name in sorted(after - before):
        print(f"{name}: not at {against}")
    digests: dict[str, set[str]] = defaultdict(set)
    for name in sorted(before & after):
        diff = Diff(digests)
        old = load(git("show", f"{against}:{folder}/{name}"))
        new = load((BASELINES / name).read_text(encoding="utf-8"))
        if len(old) != len(new):
            diff.problems.append(f"{len(old)} record(s) -> {len(new)}")
        else:
            diff.walk(old, new, "record")
        print(f"{name}: {diff.images} image(s) changed, {len(diff.problems)} other difference(s)")
        for problem in diff.problems[:20]:
            print(f"  {problem}")
        ok = ok and not diff.problems
    shas = defaultdict(set)
    for sha, new in sorted(digests.items()):
        if len(new) > 1:
            print(f"PNG {sha} became {len(new)} different digests: {sorted(new)}")
            ok = False
        for digest in new:
            shas[digest].add(sha)
    print(f"{len(digests)} distinct PNG(s) became {len(shas)} distinct pixel digest(s)")
    for digest, same in sorted(shas.items()):
        if len(same) > 1:
            print(f"  {len(same)} PNGs have the same pixels {digest}: {sorted(same)}")
    # A field baseline.Normalizer does not know would keep the PNG SHA-256 on both sides.
    for name in sorted(after):
        text = (BASELINES / name).read_text(encoding="utf-8")
        kept = digests.keys() & set(PNG_SHA256.findall(text))
        if kept:
            print(f"{name}: still holds {len(kept)} PNG SHA-256(s), e.g. {min(kept)}")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--against", default="HEAD", help="the revision to compare with")
    parser.add_argument("--check-only", action="store_true", help="compare without recording")
    args = parser.parse_args()
    failed = [] if args.check_only else record(record_modules())
    for name in failed:
        print(f"recording failed: {name}")
    ok = check(args.against) and not failed
    print("only image digests changed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
