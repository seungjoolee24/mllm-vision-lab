"""Install pinned GPU packages in an isolated Colab venv, without replacing its kernel."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", default="/content/visionlab-venv")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    env = Path(args.venv)
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    signature = {"torch": "2.9.1", "torchvision": "0.24.1", "cuda_wheels": "cu126",
                 "pyproject": (repo / "pyproject.toml").read_text(encoding="utf-8")}
    marker = env / "visionlab-install.json"
    installed = json.loads(marker.read_text()) if marker.exists() else None
    if installed != signature:
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "torch==2.9.1", "torchvision==0.24.1",
                        "--index-url", "https://download.pytorch.org/whl/cu126"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-e", f"{repo}[research]"], check=True)
        marker.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    subprocess.run([str(python), "-m", "visionlab", "doctor"], cwd=repo, check=True)
    print("Experiment Python:", python)


if __name__ == "__main__":
    main()
