"""Small, dependency-free experiment persistence primitives."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def group_id(ids):
    return digest(list(ids))[:20]


def batch_seed(base_seed, ids):
    return base_seed + sum(ids)


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    sources = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
               for p in sorted((root / "visionlab").glob("*.py"))}
    sources["pyproject.toml"] = (root / "pyproject.toml").read_text(encoding="utf-8")
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                                           text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"source_sha256": digest(sources), "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain"))}


class RunStore:
    def __init__(self, root, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id):
            raise ValueError("run-id must use only letters, numbers, _, - and .")
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path / "RUNNING.lock"

    @contextmanager
    def lock(self):
        owner = {"owner": uuid.uuid4().hex, "host": socket.gethostname(),
                 "pid": os.getpid(), "started_at": now()}
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise RuntimeError(f"Run is locked: {self.lock_path}. Stop the other runtime first. "
                               "After an interrupted runtime, use the explicit unlock command.") from None
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(owner, stream)
            yield
        finally:
            if self.lock_path.exists() and read_json(self.lock_path).get("owner") == owner["owner"]:
                self.lock_path.unlink()

    def initialize(self, manifest, groups):
        path = self.path / "manifest.json"
        if path.exists():
            previous = read_json(path)
            if previous["fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Run configuration, code or questions changed. Choose a NEW run-id.")
            if previous["runtime_key"] != manifest["runtime_key"]:
                raise ValueError("GPU, dtype or package versions changed. Choose a NEW run-id.")
        else:
            atomic_json(path, manifest)
        # A crash after the manifest but before this write is recoverable.
        if not (self.path / "batches.json").exists():
            atomic_json(self.path / "batches.json", groups)

    def save_batch(self, ids, value):
        assert value["task_ids"] == list(ids)
        atomic_json(self.path / "batches" / f"{group_id(ids)}.json", value)

    def completed(self):
        return [read_json(p) for p in sorted((self.path / "batches").glob("*.json"))]

    def state(self, stage, **extra):
        ids = [task for batch in self.completed() for task in batch["task_ids"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate task IDs in checkpoints")
        value = {"stage": stage, "completed": len(ids), "updated_at": now(), **extra}
        atomic_json(self.path / "state.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)


def validate_config(config):
    if not config.get("model_revision") or config["model_revision"] in {"main", "master"}:
        raise ValueError("Pin model_revision to an immutable Hugging Face commit.")
    if config["batch_size"] < 1 or config["max_new_tokens"] < 1:
        raise ValueError("Batch size and output token limit must be positive")
    conditions = config["conditions"]
    if not conditions or len({x["name"] for x in conditions}) != len(conditions):
        raise ValueError("Condition names must be unique")
    if any(not re.fullmatch(r"[a-z0-9_-]+", x["name"]) or x["spatial_scale"] < 1 for x in conditions):
        raise ValueError("Use simple condition names and spatial_scale >= 1")
    if config["dtype"] not in {"auto", "bfloat16", "float16"}:
        raise ValueError("Unsupported dtype")
    return config
