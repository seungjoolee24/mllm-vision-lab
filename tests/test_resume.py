"""Exercise interrupted/OOM runs; no model downloads or GPU needed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from visionlab.core import RunStore, read_json
from visionlab.runner import execute

REPO = Path(__file__).resolve().parents[1]


class FakeBackend:
    calls = []
    interrupt_once = False
    runtime = {"gpu": "TEST ONLY"}
    runtime_key = runtime

    def __init__(self, config, dataset, model_path=None):
        pass

    def attempt(self, questions, condition, seed):
        ids = [q["taskId"] for q in questions]
        self.calls.append((condition["name"], ids, seed))
        if condition["name"] == "dense" and len(ids) > 2:
            return None, "simulated OOM"
        if ids == [3, 4] and self.interrupt_once:
            type(self).interrupt_once = False
            raise KeyboardInterrupt()
        return [{"task_id": task, "response": "TEST ONLY"} for task in ids], None


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = read_json(REPO / "configs/smoke.json")
        self.config["batch_size"] = 4
        self.questions = [{"taskId": n, "question": "test input"} for n in range(1, 5)]
        self.keys = [{"taskId": n, "ground_truth": "A"} for n in range(1, 5)]
        FakeBackend.calls = []
        FakeBackend.interrupt_once = False

    def tearDown(self):
        self.tmp.cleanup()

    def run_job(self, config=None):
        return execute(config or self.config, self.questions, self.keys, self.tmp.name,
                       self.tmp.name, "test", FakeBackend)

    def test_oom_splits_both_conditions_and_resume_keeps_the_plan(self):
        FakeBackend.interrupt_once = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_job()
        run = Path(self.tmp.name) / "test"
        self.assertEqual(read_json(run / "batches.json"), [[1, 2], [3, 4]])
        self.assertEqual(len(list((run / "batches").glob("*.json"))), 1)
        self.assertFalse((run / "RUNNING.lock").exists())
        FakeBackend.calls = []
        self.run_job()
        self.assertEqual(FakeBackend.calls, [("baseline", [3, 4], 49), ("dense", [3, 4], 49)])
        completed = [read_json(p) for p in (run / "batches").glob("*.json")]
        self.assertEqual(sorted(task for batch in completed for task in batch["task_ids"]), [1, 2, 3, 4])
        FakeBackend.calls = []
        self.run_job()
        self.assertEqual(FakeBackend.calls, [])

    def test_changed_prompt_cannot_silently_resume(self):
        self.run_job()
        changed = copy.deepcopy(self.config)
        changed["prompt_suffix"] = "different treatment"
        with self.assertRaisesRegex(ValueError, "NEW run-id"):
            self.run_job(changed)

    def test_concurrent_run_is_rejected(self):
        store = RunStore(self.tmp.name, "test")
        with store.lock():
            with self.assertRaisesRegex(RuntimeError, "locked"):
                self.run_job()

    def test_runtime_change_cannot_silently_resume_partial_run(self):
        FakeBackend.interrupt_once = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_job()
        previous = FakeBackend.runtime_key
        try:
            FakeBackend.runtime_key = {"gpu": "DIFFERENT TEST GPU"}
            with self.assertRaisesRegex(ValueError, "GPU"):
                self.run_job()
        finally:
            FakeBackend.runtime_key = previous


if __name__ == "__main__":
    unittest.main()
