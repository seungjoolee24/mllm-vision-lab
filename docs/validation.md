# 검증 기록

구축일: 2026-11-02, Asia/Seoul.

## 확인한 항목

- CPU persistence tests: 메모리 부족 시 모든 조건 재실행, 중간 종료 후 같은 분할 계획으로 재개, 완료 배치 중복 계산 방지, 다른 프롬프트/환경의 재개 거부, 중복 실행 lock.
- Repository validator: config 구조, Colab notebook 코드 구문, 중첩 askpass Python 구문, 저장 출력 제거, 기존 HTML 이미지·파일 연결, 코드/노트북의 credential 패턴 검사.
- 로컬 RTX 4060 Ti 8GB, BF16, 기존 Qwen3-VL-2B-Instruct 가중치로 새 공통 실행기의 2문항 × 2조건 실제 추론 및 전체 질문·이미지·원문 보고서 생성.
- 모델·데이터 원본은 기존에 검증한 commit과 SHA256를 사용.

## 계정·원격 런타임 확인의 구분

로컬 GPU 검증은 Colab GPU 검증을 대신하지 않습니다. GitHub 원격 push, Google Drive 마운트, Colab의 실제 GPU 배정·격리 venv 설치·2문항 추론은 해당 계정 인증과 원격 실행 결과로 별도로 확인해야 합니다.

Colab 노트북은 위 원격 smoke를 실행하도록 준비되어 있습니다. 실행 전에 결과가 확인됐다고 해석하지 마세요. 전체 388문항 실험은 자동으로 시작하지 않습니다.
