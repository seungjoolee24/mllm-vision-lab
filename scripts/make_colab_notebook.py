"""Generate a clean Colab entry point with no credentials or saved outputs."""
import json
from pathlib import Path
import textwrap

cells = []


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    cells.append(value)


cell("markdown", """
    # MLLM Vision Lab · BabyVision

    GitHub는 코드·노트북·설정, Colab은 GPU, Google Drive는 결과·재개 기록을 담당합니다.
    데스크탑과 노트북에서 같은 Google/GitHub 계정으로 이 노트북을 열면 됩니다.

    1. **런타임 → 런타임 유형 변경 → GPU**를 선택하세요.
    2. 비공개 저장소는 Colab의 GitHub 열기 화면에서 **Include private repos**를 켜서 여세요.
    3. 왼쪽 **Secrets(열쇠)**에 `GITHUB_TOKEN`을 추가하세요. 이 저장소의 **Contents: Read-only** 권한인
       fine-grained GitHub token이면 clone에 충분합니다. 노트북 접근을 허용하세요.
       Colab에서 노트북을 여는 권한과 런타임의 `git clone` 인증은 별개입니다.
    4. 우선 기본값 `smoke.json`을 실행하세요. 2문항·64 출력 토큰의 인프라 확인이며 연구 점수가 아닙니다.
    5. 확인 후 `pilot52.json` 또는 `full388.json`과 **새 RUN_ID**를 사용하세요.

    같은 RUN_ID를 두 런타임에서 동시에 실행하지 마세요. 기기 전환 시 기존 실행을 중지한 뒤 이어가세요.
    실험이 끝나지 않은 상태에서 **런타임 연결 해제 및 삭제**를 하면 현재 배치는 다시 계산될 수 있습니다.
""")
cell("code", '''
    # 실행 설정: 재개할 때 RUN_ID, CONFIG, CODE_REF를 그대로 유지하세요.
    REPO_SLUG = "seungjoolee24/mllm-vision-lab" #@param {type:"string"}
    CODE_REF = "main" # main 또는 고정된 40자리 Git commit SHA
    CONFIG = "configs/smoke.json" #@param ["configs/smoke.json", "configs/pilot52.json", "configs/full388.json"]
    RUN_ID = "smoke-v1" #@param {type:"string"}
    DTYPE_OVERRIDE = "" #@param ["", "bfloat16", "float16"]
    # pilot/full은 BF16 조건입니다. T4 등에서 FP16을 선택하면 새 RUN_ID를 쓰세요.
''')
cell("markdown", """
    ## 1. GitHub 코드 받기
    토큰은 Colab Secrets 또는 숨김 입력으로만 사용합니다. URL·파일·출력에 저장하지 않습니다.
    기존 checkout에 수정 내용이 있으면 자동 덮어쓰기를 중단합니다.
""")
cell("code", '''
    import getpass, os, re, subprocess, sys, tempfile
    from pathlib import Path
    from google.colab import userdata

    assert re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", REPO_SLUG)
    assert CODE_REF == "main" or re.fullmatch(r"[0-9a-f]{40}", CODE_REF)
    REPO = Path("/content/mllm-vision-lab")
    try:
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        token = getpass.getpass("GitHub read-only token (hidden; public repo can be blank): ")
    try:
        with tempfile.TemporaryDirectory(prefix="visionlab-git-") as temporary:
            askpass = Path(temporary) / "askpass.py"
            askpass.write_text("#!/usr/bin/env python3\\nimport os,sys\\nprint('x-access-token' if 'username' in sys.argv[1].lower() else os.environ.get('VISIONLAB_GIT_TOKEN',''))\\n")
            askpass.chmod(0o700)
            git_env = dict(os.environ, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT="0")
            git_env["VISIONLAB_GIT_TOKEN"] = token or ""
            url = "https://github.com/" + REPO_SLUG + ".git"
            if not REPO.exists():
                subprocess.run(["git", "-c", "credential.helper=", "clone", "--branch", "main", url, str(REPO)], env=git_env, check=True)
            else:
                existing = subprocess.check_output(["git", "-C", str(REPO), "remote", "get-url", "origin"], text=True).strip()
                assert existing == url, "Existing checkout belongs to another repository"
                dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
                assert not dirty, "Commit/push or save your local edits before refreshing this checkout"
                subprocess.run(["git", "-C", str(REPO), "-c", "credential.helper=", "fetch", "origin", "main"], env=git_env, check=True)
            if CODE_REF == "main":
                subprocess.run(["git", "-C", str(REPO), "checkout", "main"], check=True)
                subprocess.run(["git", "-C", str(REPO), "merge", "--ff-only", "origin/main"], check=True)
            else:
                subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", CODE_REF], check=True)
            git_env.pop("VISIONLAB_GIT_TOKEN", None)
    finally:
        token = None
        if "git_env" in globals():
            git_env.pop("VISIONLAB_GIT_TOKEN", None)
    print("Code commit:", subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())
''')
cell("markdown", """
    ## 2. 지속 저장소와 임시 캐시
    Drive에는 결과·배치 체크포인트·보고서가 저장됩니다. 모델·데이터 캐시는 Colab VM에 두어 읽기 속도를 확보합니다.
    새 VM에서는 캐시를 다시 받지만, 완료한 배치는 Drive에서 복구합니다.
""")
cell("code", '''
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_HOME = Path("/content/drive/MyDrive/mllm-vision-lab")
    PROJECT_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["MLLM_RUNS"] = str(PROJECT_HOME / "runs")
    os.environ["MLLM_DATA_CACHE"] = "/content/mllm-cache/babyvision"
    os.environ["HF_HOME"] = "/content/mllm-cache/huggingface"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"
    print("Persistent experiment:", PROJECT_HOME / "runs" / RUN_ID)
''')
cell("markdown", """
    ## 3. 격리된 GPU 환경 설치 및 확인
    PyTorch 2.9.1 + CUDA 12.6 wheel과 Transformers 4.57.6을 별도 venv에 설치합니다.
    Colab 커널의 패키지를 교체하지 않습니다. 최초 설치와 모델 다운로드에는 시간이 걸립니다.
""")
cell("code", '''
    subprocess.run([sys.executable, str(REPO / "scripts/bootstrap_colab.py")], cwd=REPO, check=True)
    PYTHON = "/content/visionlab-venv/bin/python"
    subprocess.run([PYTHON, "-m", "visionlab", "doctor"], cwd=REPO, check=True)
''')
cell("markdown", """
    ## 4. 실행 또는 재개
    같은 RUN_ID의 완료 배치는 건너뜁니다. 설정·코드·GPU·정밀도·주요 패키지가 바뀌면 새 RUN_ID가 필요합니다.
    메모리 부족 시 배치를 나누고, 모든 입력 조건을 같은 하위 배치·시드로 다시 계산합니다.
    기존 로컬 52문항과 같은 문제를 선택할 수 있지만, 배치 구성이 달라 이전 실행을 정확히 재현하는 것은 아닙니다.
""")
cell("code", '''
    import json
    cfg = json.loads((REPO / CONFIG).read_text())
    print(json.dumps({"run_id": RUN_ID, "config": cfg, "dtype_override": DTYPE_OVERRIDE or None}, indent=2))
    command = [PYTHON, "-m", "visionlab", "run", "--config", CONFIG, "--run-id", RUN_ID]
    if DTYPE_OVERRIDE:
        command += ["--dtype", DTYPE_OVERRIDE]
    subprocess.run(command, cwd=REPO, check=True)
''')
cell("markdown", """
    ## 5. 질문·이미지·답변을 직접 확인
    아래는 처음 3문항입니다. 모든 문항의 원문과 정답은 Drive의 `runs/RUN_ID/report/`에 있습니다.
    보고서는 boxed 정규화 일치만 집계합니다. 이전 로컬의 수동 의미 검토 점수와 채점 기준이 다릅니다.
""")
cell("code", '''
    from html import escape
    from IPython.display import display, HTML, Image
    run_path = Path(os.environ["MLLM_RUNS"]) / RUN_ID
    results = json.loads((run_path / "results.json").read_text())
    display(HTML("<h3>집계</h3><pre>" + escape(json.dumps(results["summary"], ensure_ascii=False, indent=2)) + "</pre>"))
    for row in results["results"][:3]:
        display(HTML("<h3>#" + str(row["taskId"]) + " · " + escape(row["question"]) + "</h3>"))
        image = next((run_path / "report/images").glob(str(row["taskId"]) + ".*"))
        display(Image(filename=str(image), width=650))
        display(HTML("<p>데이터셋 정답: <b>" + escape(row["ground_truth"]) + "</b></p>"))
        for name, prediction in row["predictions"].items():
            display(HTML("<h4>" + escape(name) + "</h4><pre>" + escape(prediction["response"]) + "</pre>"))
    print("All evidence:", run_path / "report/index.html")
''')
cell("markdown", """
    ## 6. 선택: 전체 결과 내려받기
    Drive에 이미 저장되어 있습니다. 아래 셀은 결과와 이미지·보고서를 ZIP으로 다운로드합니다.
    압축을 푼 뒤 `report/index.html`을 여세요.
""")
cell("code", '''
    DOWNLOAD_RESULTS = False #@param {type:"boolean"}
    if DOWNLOAD_RESULTS:
        import shutil
        from google.colab import files
        bundle = shutil.make_archive("/content/" + RUN_ID, "zip", run_path)
        files.download(bundle)
''')
cell("markdown", """
    ## 다른 기기에서 이어가기
    같은 GitHub 노트북을 열고 같은 Google Drive 계정, RUN_ID, CONFIG, CODE_REF를 사용하세요.
    진행 중인 실험은 `manifest.json`의 `code.git_commit` 값을 CODE_REF로 고정해 재개하는 편이 안전합니다.
    GPU/정밀도/패키지가 바뀌면 새 실행으로 분리됩니다. 원래 VM이 완전히 종료됐는데 lock이 남았을 때만
    아래 명령을 **별도 코드 셀**에서 실행하세요.

    ```python
    subprocess.run([PYTHON, "-m", "visionlab", "unlock", "--run-id", RUN_ID,
                    "--old-runtime-is-stopped"], cwd=REPO, check=True)
    ```

    **코드 변경 저장:** 노트북 변경은 File → Save a copy in GitHub로 저장하세요.
    런타임에서 수정한 `.py` 파일은 자동으로 GitHub에 저장되지 않습니다. Git으로 commit/push하거나
    GitHub 웹 편집기에서 반영하세요. 실험 중 소스 수정은 새 RUN_ID에서 실행하세요.
""")

notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"name": "babyvision_colab.ipynb", "provenance": []},
                         "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}}}
for i, c in enumerate(cells):
    c["id"] = f"visionlab-{i:02d}"
    c["metadata"]["id"] = c["id"]
out = Path(__file__).resolve().parents[1] / "notebooks/babyvision_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(out)
