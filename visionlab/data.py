"""Pinned BabyVision download; answer keys stay outside inference inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
from .core import atomic_json, digest, read_json

SOURCE = {
    "repository": "https://github.com/UniPat-AI/BabyVision",
    "commit": "7f92fd4b1dc1c68b7b936a9bc09c68b4a944a55a",
    "archive_url": "https://raw.githubusercontent.com/UniPat-AI/BabyVision/7f92fd4b1dc1c68b7b936a9bc09c68b4a944a55a/data/babyvision_data.zip",
    "archive_sha256": "e5e16e7104b080bac7e67bb98d74bd5057f7151f487cd2f61991106253751e54",
    "metadata_sha256": "9fae5778235c61f6806f6e5c848b6913195464fb8f7bc816f8f2bd71beec5584",
}
INPUT_FIELDS = ("taskId", "type", "subtype", "image", "question", "ansType", "options")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "babyvision_data.zip"
    if not archive.exists():
        tmp = archive.with_suffix(".zip.part")
        request = urllib.request.Request(SOURCE["archive_url"], headers={"User-Agent": "mllm-vision-lab"})
        with urllib.request.urlopen(request, timeout=60) as source, tmp.open("wb") as output:
            shutil.copyfileobj(source, output)
        if sha256_file(tmp) != SOURCE["archive_sha256"]:
            raise ValueError("BabyVision archive checksum mismatch")
        tmp.replace(archive)
    if sha256_file(archive) != SOURCE["archive_sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    with zipfile.ZipFile(archive) as z:
        base = cache.resolve()
        for item in z.infolist():
            target = (cache / item.filename).resolve()
            if not target.is_relative_to(base):
                raise ValueError("Unsafe dataset archive path")
        marker = cache / "source-manifest.json"
        expected = {**SOURCE, "archive_bytes": archive.stat().st_size}
        if not marker.exists() or read_json(marker) != expected or not (cache / "babyvision_data/meta_data.jsonl").exists():
            z.extractall(cache)
            atomic_json(marker, expected)
    return cache / "babyvision_data"


def prepare(dataset, selection, repo_root):
    dataset = Path(dataset)
    if sha256_file(dataset / "meta_data.jsonl") != SOURCE["metadata_sha256"]:
        raise ValueError("BabyVision metadata differs from the pinned official version")
    all_rows = [json.loads(line) for line in (dataset / "meta_data.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {int(row["taskId"]): row for row in all_rows}
    if len(by_id) != 388 or len(all_rows) != 388:
        raise ValueError("Unexpected BabyVision dataset size")
    if selection == "all":
        ids = sorted(by_id)
    else:
        manifest = read_json(Path(repo_root) / selection)
        ids = manifest["task_ids"]
    if len(ids) != len(set(ids)) or not set(ids).issubset(by_id):
        raise ValueError("Selection contains duplicate or missing task IDs")
    questions, keys = [], []
    for task in ids:
        row = by_id[task]
        image = (dataset / row["image"]).resolve()
        if not image.is_relative_to(dataset.resolve()) or not image.is_file():
            raise ValueError(f"Missing or unsafe image for {task}")
        question = {key: row[key] for key in INPUT_FIELDS}
        question["image_sha256"] = sha256_file(image)
        questions.append(question)
        truth = row["blankAns"] if row["ansType"] == "blank" else chr(65 + int(row["choiceAns"]))
        keys.append({"taskId": task, "ground_truth": str(truth)})
    return questions, keys, digest(questions)


def make_prompt(question, suffix):
    text = question["question"]
    if question["ansType"] != "blank":
        text += "\nChoices:\n" + "\n".join(f"({chr(65+i)}) {choice}" for i, choice in enumerate(question["options"]))
    return text + suffix
