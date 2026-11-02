import argparse
import json
import os
from pathlib import Path
from .core import RunStore, read_json, validate_config
from .data import ensure_dataset, prepare


def main():
    parser = argparse.ArgumentParser(prog="visionlab")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Print GPU, package and precision information")
    data = commands.add_parser("download-data", help="Download and verify the pinned official archive")
    data.add_argument("--cache", default=os.getenv("MLLM_DATA_CACHE", "data/babyvision"))
    run = commands.add_parser("run", help="Run or resume an experiment")
    run.add_argument("--config", default="configs/smoke.json")
    run.add_argument("--run-id", required=True)
    run.add_argument("--runs-root", default=os.getenv("MLLM_RUNS", "runs"))
    run.add_argument("--dataset", help="Existing verified babyvision_data directory")
    run.add_argument("--cache", default=os.getenv("MLLM_DATA_CACHE", "data/babyvision"))
    run.add_argument("--model-path", help="Existing local model; its weights are hashed in the runtime key")
    run.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], help="Explicit new experimental condition")
    run.add_argument("--batch-size", type=int, help="Explicitly override the planned batch size")
    report = commands.add_parser("report", help="Regenerate all question/answer evidence")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--dataset", required=True)
    unlock = commands.add_parser("unlock", help="Remove a lock ONLY after the old runtime stopped")
    unlock.add_argument("--run-id", required=True)
    unlock.add_argument("--runs-root", default=os.getenv("MLLM_RUNS", "runs"))
    unlock.add_argument("--old-runtime-is-stopped", action="store_true", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.command == "doctor":
        from .qwen import runtime_info
        print(json.dumps(runtime_info(), ensure_ascii=False, indent=2))
    elif args.command == "download-data":
        print(ensure_dataset(args.cache))
    elif args.command == "report":
        from .report import build
        print(build(args.run_dir, args.dataset))
    elif args.command == "unlock":
        store = RunStore(args.runs_root, args.run_id)
        if store.lock_path.exists():
            print("Removing previous owner lock:", json.dumps(read_json(store.lock_path)))
            store.lock_path.unlink()
    elif args.command == "run":
        config_path = Path(args.config)
        if not config_path.exists():
            config_path = repo / config_path
        config = read_json(config_path)
        if args.dtype:
            config["dtype"] = args.dtype
        if args.batch_size:
            config["batch_size"] = args.batch_size
        validate_config(config)
        dataset = Path(args.dataset) if args.dataset else ensure_dataset(args.cache)
        questions, keys, _ = prepare(dataset, config["selection"], repo)
        from .qwen import QwenBackend
        from .runner import execute
        run_path = execute(config, questions, keys, dataset, args.runs_root, args.run_id,
                           QwenBackend, model_path=args.model_path)
        from .report import build
        print("Report:", build(run_path, dataset))


if __name__ == "__main__":
    main()
