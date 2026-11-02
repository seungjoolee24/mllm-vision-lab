# MLLM Vision Lab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seungjoolee24/mllm-vision-lab/blob/main/notebooks/babyvision_colab.ipynb)

BabyVision의 세밀한 시각 문제를 Qwen3-VL로 진단하는 개인 연구 환경입니다. 코드와 실험 설정은 GitHub에서 관리하고, GPU 연산은 Colab에서, 실행 결과와 재개 기록은 Google Drive에서 유지합니다. 데스크탑과 노트북 모두 브라우저만으로 원격 실험을 실행할 수 있습니다.

| 위치 | 저장하는 내용 |
|---|---|
| 이 GitHub 저장소 | Python 코드, Colab 노트북, 고정 설정, 문항 ID |
| Colab VM | GPU 프로세스, 격리된 Python venv, 모델·데이터 임시 캐시 |
| Google Drive `MyDrive/mllm-vision-lab/runs/` | 완료 배치, 실험 설정·환경, 질문·정답, 보고서·이미지 |
| 각 PC의 checkout | 선택 사항. 코드 수정은 commit/push 후 다른 기기에서 pull |

## 첫 실행

1. 위 **Open in Colab**을 여세요. 현재 공개 저장소이므로 GitHub 인증이나 토큰 없이 열고 코드를 받을 수 있습니다.
2. Colab 런타임을 GPU로 바꾸세요. 우선 기본 `configs/smoke.json`, `RUN_ID="smoke-v1"`을 사용하세요.
3. 공개 저장소에서는 `REPO_IS_PRIVATE=False`를 유지하세요. **나중에 비공개로 바꿀 때만** [Colab GitHub 열기](https://colab.research.google.com/github)에서 Include private repos를 허용하고 `REPO_IS_PRIVATE=True`를 사용하세요. 비공개 런타임 clone은 [fine-grained token](https://github.com/settings/personal-access-tokens/new)의 Only selected repositories → 이 저장소, Contents: Read-only를 지정해 Colab Secrets의 `GITHUB_TOKEN`으로 저장하면 됩니다. 토큰은 코드나 채팅에 붙여 넣지 마세요.
4. 셀을 위에서 순서대로 실행하고, 본인 Google Drive 마운트를 허용하세요.
5. GPU 정보, 2문항의 양쪽 입력 결과, 보고서 저장 위치를 확인하세요. 이 smoke는 출력 상한 64토큰으로 **설치·실행 확인용**입니다.
6. 연구 실행은 `configs/pilot52.json` 또는 `configs/full388.json`과 새 RUN_ID를 선택하세요. 이 두 설정의 출력 상한은 512토큰입니다.

Google 계정 로그인과 Drive 권한 승인은 처음에 사용자 본인이 해야 합니다. 공개 저장소를 읽는 데에는 GitHub 로그인이 필요 없습니다. Colab의 유료 GPU 선택·구독은 이 코드가 자동으로 구매하지 않습니다.

## 설정

| 설정 | 문항 | 최초 배치 | 정밀도 | 조건 | 용도 |
|---|---:|---:|---|---|---|
| `configs/smoke.json` | 2 | 2 | GPU에 따라 BF16/FP16, 실제 값 기록 | 기준 / 약 2배 시각 토큰 | 인프라 확인, 64 출력 토큰 |
| `configs/pilot52.json` | 기존과 같은 52 | 12 | BF16 | 기준 / 약 2배 시각 토큰 | 512 출력 토큰의 새 대응 비교 |
| `configs/full388.json` | 전체 388 | 12 | BF16 | 기준 / 약 2배 시각 토큰 | 전체 데이터 진단 |

모델은 `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`입니다. 다른 모델은 config 사본에서 ID와 고정 revision을 함께 바꾸고 새 RUN_ID로 실행하세요. 현재 모델 로더와 해상도 제어는 Qwen3-VL 구조용입니다.

pilot/full에서 BF16을 지원하지 않는 GPU가 배정되면 명시적으로 중단합니다. FP16으로 진행하려면 노트북의 `DTYPE_OVERRIDE="float16"`과 새 RUN_ID를 사용하세요. 모든 조건을 같은 GPU·정밀도에서 비교하세요.

## 다른 PC에서 재개

1. 같은 공개 GitHub 노트북을 여세요. Drive는 기존 결과가 있는 **같은 Google 계정**으로 연결하세요.
2. 기존 런타임을 먼저 중지하세요. 같은 RUN_ID를 두 런타임에서 동시에 실행하지 마세요.
3. 기존 RUN_ID, CONFIG, 정밀도를 유지하세요. `manifest.json`의 `code.git_commit` 값을 `CODE_REF`에 넣으면 같은 코드를 선택할 수 있습니다.
4. 같은 실행 셀을 실행하세요. 저장된 배치 계획과 완료 배치를 읽고 이어갑니다. 중단된 현재 배치만 다시 실행될 수 있습니다.
5. GPU, 정밀도, 주요 패키지 또는 코드·설정이 달라졌다면 새 RUN_ID로 분리하세요.

Drive lock은 중복 실행의 실수 방지 장치입니다. 여러 VM에 걸친 분산 잠금 서버를 대신하지 않습니다. VM 강제 종료로 lock이 남았을 때만 기존 VM 종료를 확인한 뒤 노트북 마지막 설명의 명시적 `unlock` 명령을 사용하세요.

노트북 수정은 **File → Save a copy in GitHub**로 저장합니다. 런타임에서 수정한 Python 소스 파일은 자동 동기화되지 않습니다. GitHub 웹 편집기 또는 git commit/push로 저장해야 다른 기기에서 받을 수 있습니다.

## PC에서 코드 수정

GPU를 사용하지 않을 노트북에는 CUDA가 필요하지 않습니다. GitHub 웹 편집기만 사용하거나 다음처럼 checkout을 받으세요.

```sh
git clone https://github.com/seungjoolee24/mllm-vision-lab.git
cd mllm-vision-lab
git pull --ff-only
git switch -c codex/my-experiment
# 코드/설정 수정
git add visionlab configs notebooks
git commit -m "Describe the experimental change"
git push -u origin codex/my-experiment
```

현재 Colab 시작 셀은 `main` 또는 고정 commit SHA를 받습니다. 브랜치 작업은 main에 반영하거나 해당 commit SHA를 `CODE_REF`에 넣으세요. CPU 테스트에는 Python 3.11~3.13이면 충분합니다.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/validate_repository.py
```

이미 구성된 데스크탑 GPU 환경에서는 기존 venv의 Python으로 저장소 루트에서 실행할 수 있습니다.

```powershell
. 'E:\mllm-research\activate-research.ps1' -Quiet
python -m visionlab doctor
python -m visionlab run --config configs/smoke.json --run-id desktop-smoke-v1 `
  --dataset E:\mllm-research\datasets\babyvision\babyvision_data `
  --model-path E:\mllm-research\models\Qwen--Qwen3-VL-2B-Instruct `
  --runs-root E:\mllm-research\experiments\portable-runs
```

## 보고서와 이전 기록

새 실행은 각 run 폴더에 `manifest.json`, `batches.json`, `state.json`, `batches/*.json`, `summary.json`, `results.json`, `report/index.html`을 만듭니다. 보고서는 모든 문항의 이미지·질문·정답·각 조건의 전체 답변을 바로 보여 줍니다. 다운로드 셀로 ZIP을 받은 뒤 압축을 풀고 `report/index.html`을 열면 됩니다.

사용자 선택에 따라 **기존 로컬 실험의 이미지·모델 답변·보고서는 GitHub에 업로드하지 않습니다.** 원래 데스크탑의 로컬 `history/local-2026-11-02/`와 기존 연구 디렉터리에 보존되어 있습니다. `history/`는 Git에서 제외됩니다. 다른 기기의 clone에는 이 기록이 포함되지 않습니다.

**새 자동 보고서는 boxed 정규화 일치만 채점합니다.** 이전 로컬 보고서는 일부 답변에 수동 의미 검토를 적용했습니다. 두 점수를 같은 채점 규칙으로 오해하지 마세요. 새 pilot52도 기존 혼합 배치를 재생하지 않고 새로운 배치 계획으로 양쪽 조건을 함께 실행합니다.

[실험 프로토콜과 해석 범위](docs/experiment-protocol.md) · [검증 기록](docs/validation.md)

## 공식 참고 자료

- [Colab과 GitHub 연동](https://github.com/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)
- [Colab 저장·리소스 제한](https://research.google.com/colaboratory/faq.html)
- [BabyVision 원본](https://github.com/UniPat-AI/BabyVision)
- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

모델·원본 데이터의 이용 조건은 각 원본 배포처를 따릅니다. 이 저장소의 결과는 공식 리더보드 점수가 아닙니다.
