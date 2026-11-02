"""Atomic paired checkpoints; OOM splits are persisted before retrying both conditions."""
from pathlib import Path
from .core import RunStore, atomic_json, batch_seed, digest, now, read_json, source_provenance, validate_config
from .data import SOURCE


def execute(config, questions, keys, dataset, runs_root, run_id, backend_factory, model_path=None):
    validate_config(config)
    ids = [q["taskId"] for q in questions]
    by_id = {q["taskId"]: q for q in questions}
    if len(ids) != len(by_id) or len(keys) != len(ids):
        raise ValueError("Invalid question/answer selection")
    provenance = source_provenance()
    fingerprint = digest({"config": config, "questions": questions, "answer_key_sha256": digest(keys), "source": provenance["source_sha256"],
                          "dataset": SOURCE["archive_sha256"]})
    store = RunStore(runs_root, run_id)
    with store.lock():
        previous_path = store.path / "manifest.json"
        if previous_path.exists() and read_json(previous_path)["fingerprint"] != fingerprint:
            raise ValueError("Run configuration, code or questions changed. Choose a NEW run-id.")
        if previous_path.exists() and (store.path / "state.json").exists():
            if read_json(store.path / "state.json")["stage"] == "completed":
                # Completed runs need no GPU, model reload or network access.
                print(f"Already completed: {store.path}", flush=True)
                return store.path
        store.state("loading_model", total=len(ids))
        try:
            backend = backend_factory(config, dataset, model_path=model_path)
            manifest = {"schema_version": 1, "created_at": now(), "fingerprint": fingerprint,
                        "config": config, "task_ids": ids, "dataset_source": SOURCE,
                        "question_sha256": digest(questions), "code": provenance,
                        "runtime": backend.runtime, "runtime_key": backend.runtime_key,
                        "protocol": "Unofficial paired diagnostic; sampling; no external LLM judge",
                        "seed_rule": "base_seed + sum(task_ids in the actual batch)",
                        "resize_rule": "baseline smart_resize dimensions times spatial_scale, nearest 32 pixels"}
            groups = [ids[i:i+config["batch_size"]] for i in range(0, len(ids), config["batch_size"])]
            store.initialize(manifest, groups)
            atomic_json(store.path / "questions.json", questions)
            atomic_json(store.path / "answer-key.json", keys)
            events_path = store.path / "resource-events.json"
            events = read_json(events_path) if events_path.exists() else []
            index = 0
            groups = read_json(store.path / "batches.json")
            while index < len(groups):
                group = groups[index]
                completed = {tuple(b["task_ids"]) for b in store.completed()}
                if tuple(group) in completed:
                    index += 1
                    continue
                seed = batch_seed(config["seed"], group)
                predictions, error = {}, None
                for condition in config["conditions"]:
                    store.state("running", total=len(ids), task_ids=group, condition=condition["name"], seed=seed)
                    records, error = backend.attempt([by_id[x] for x in group], condition, seed)
                    if error:
                        events.append({"at": now(), "event": "oom", "task_ids": group,
                                       "condition": condition, "message": error})
                        atomic_json(events_path, events)
                        break
                    if [r["task_id"] for r in records] != group:
                        raise ValueError("Backend result order changed")
                    predictions[condition["name"]] = records
                if error:
                    if len(group) == 1:
                        raise RuntimeError(f"Single image {group} exceeds VRAM. Choose a lower-resolution NEW run.")
                    middle = (len(group) + 1) // 2
                    groups[index:index+1] = [group[:middle], group[middle:]]
                    atomic_json(store.path / "batches.json", groups)
                    store.state("batch_split", total=len(ids), failed_task_ids=group,
                                retry="Both/all conditions rerun on each identical subgroup")
                    continue
                store.save_batch(group, {"task_ids": group, "seed": seed, "conditions": predictions,
                                         "completed_at": now()})
                index += 1
            final_ids = [x for b in store.completed() for x in b["task_ids"]]
            if len(final_ids) != len(ids) or set(final_ids) != set(ids):
                raise RuntimeError("Checkpoint coverage is incomplete")
            store.state("completed", total=len(ids))
        except BaseException as error:
            store.state("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        total=len(ids), error=f"{type(error).__name__}: {error}")
            raise
    return store.path
