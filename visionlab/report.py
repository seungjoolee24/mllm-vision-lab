"""Question-first HTML evidence and explicitly unofficial strict scores."""
from collections import Counter
from html import escape
import json
from pathlib import Path
import re
import shutil
import unicodedata
from .core import atomic_json, read_json


def normalize(value):
    value = unicodedata.normalize("NFKC", str(value or "")).strip()
    value = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", value)
    return re.sub(r"\s+", "", value.replace("\\left", "").replace("\\right", "").replace("$", "")).casefold().rstrip(".。")


def build(run, dataset):
    run, dataset = Path(run), Path(dataset)
    manifest = read_json(run / "manifest.json")
    questions = read_json(run / "questions.json")
    keys = {r["taskId"]: r["ground_truth"] for r in read_json(run / "answer-key.json")}
    names = [x["name"] for x in manifest["config"]["conditions"]]
    batches = [read_json(p) for p in sorted((run / "batches").glob("*.json"))]
    by_id = {name: {} for name in names}
    timings = {name: 0.0 for name in names}
    for batch in batches:
        for name in names:
            records = batch["conditions"][name]
            timings[name] += records[0]["batch_seconds"]
            for row in records:
                if row["task_id"] in by_id[name]:
                    raise ValueError("Duplicate prediction")
                by_id[name][row["task_id"]] = row
    out = run / "report"
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows, cards = [], []
    summary = {"planned": len(questions), "completed_pairs": 0, "conditions": {},
               "official_leaderboard_comparable": False, "scoring": "Normalized boxed exact match only"}
    for name in names:
        predictions = list(by_id[name].values())
        summary["conditions"][name] = {"completed": len(predictions),
            "strict_correct": sum(p["extracted_answer"] is not None and normalize(p["extracted_answer"]) == normalize(keys[p["task_id"]]) for p in predictions),
            "missing_boxed": sum(p["extracted_answer"] is None for p in predictions),
            "token_limit_hits": sum(p["hit_token_limit"] for p in predictions),
            "generation_seconds": round(timings[name], 3),
            "peak_allocated_gib": max((p["peak_allocated_gib"] for p in predictions), default=0)}
    for q in questions:
        task = q["taskId"]
        if not all(task in by_id[name] for name in names):
            continue
        summary["completed_pairs"] += 1
        image_name = f"{task}{Path(q['image']).suffix.lower()}"
        shutil.copyfile(dataset / q["image"], images / image_name)
        preds = {}
        panels = []
        for name in names:
            p = by_id[name][task]
            correct = p["extracted_answer"] is not None and normalize(p["extracted_answer"]) == normalize(keys[task])
            preds[name] = {**p, "strict_correct": correct}
            answer = p["extracted_answer"] if p["extracted_answer"] is not None else "최종 boxed 답 없음"
            panels.append(f"<div><h3>{escape(name)} · {'정규화 일치' if correct else '불일치/미추출'}</h3>"
                          f"<p>추출 답: <b>{escape(str(answer))}</b> · 시각 토큰 {p['visual_tokens']} · "
                          f"입력 크기 {escape(str(p['processed_image_size']))} · 출력 {p['output_tokens']}토큰</p>"
                          f"<details open><summary>모델 답변 전체</summary><pre>{escape(p['response'])}</pre></details></div>")
        rows.append({**q, "ground_truth": keys[task], "predictions": preds})
        choices = " ".join(f"({chr(65+i)}) {choice}" for i, choice in enumerate(q["options"]))
        cards.append(f"<article id='q-{task}'><h2>#{task} · {escape(q['type'])} / {escape(q['subtype'])}</h2>"
                     f"<div class='question'><img loading='lazy' src='images/{image_name}' alt='문제 {task}'>"
                     f"<div><h3>질문 원문</h3><p>{escape(q['question'])}</p><p>{escape(choices)}</p>"
                     f"<p><b>데이터셋 정답: {escape(keys[task])}</b></p></div></div>"
                     f"<div class='answers'>{''.join(panels)}</div></article>")
    if len(names) == 2:
        transitions = Counter((r["predictions"][names[0]]["strict_correct"], r["predictions"][names[1]]["strict_correct"]) for r in rows)
        summary["paired_transitions"] = {"both_correct": transitions[True, True], "improved": transitions[False, True],
                                         "regressed": transitions[True, False], "both_wrong": transitions[False, False]}
    category_rows = []
    summary["categories"] = {}
    for category in dict.fromkeys(q["type"] for q in questions):
        group = [r for r in rows if r["type"] == category]
        counts = {n: sum(r["predictions"][n]["strict_correct"] for r in group) for n in names}
        summary["categories"][category] = {"completed": len(group), "strict_correct": counts}
        category_rows.append("<tr><td>"+escape(category)+"</td>"+"".join(f"<td>{counts[n]}/{len(group)}</td>" for n in names)+"</tr>")
    scores = "".join(f"<tr><td>{escape(n)}</td><td>{v['strict_correct']}/{v['completed']}</td><td>{v['missing_boxed']}</td>"
                     f"<td>{v['token_limit_hits']}</td><td>{v['generation_seconds']}</td><td>{v['peak_allocated_gib']}</td></tr>"
                     for n, v in summary["conditions"].items())
    css = "body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#21334a;margin:0}main{max-width:1200px;margin:auto;padding:24px}header,article,section{background:white;padding:24px;margin:20px 0;border:1px solid #d5deeb;border-radius:10px}.question,.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:22px}img{width:100%;max-height:520px;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:14px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;text-align:left;padding:8px}.scroll{overflow-x:auto}summary{font-weight:bold;cursor:pointer}"
    html = f"<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>BabyVision 실험 보고서</title><style>{css}</style><main><header><h1>BabyVision · {escape(run.name)}</h1><p>완료 {summary['completed_pairs']}/{len(questions)}문항 · {escape(manifest['config']['model_id'])}</p><p>비공식 진단입니다. 정규화한 boxed 답만 채점했으며, 형식 없는 정답이나 의미 동등성은 별도 검토가 필요합니다. 이전 로컬 보고서의 수동 의미 검토 점수와 직접 비교하지 마세요.</p><div class='scroll'><table><tr><th>조건</th><th>정규화 정답</th><th>boxed 누락</th><th>출력 상한</th><th>생성 시간(초)</th><th>최대 할당 GiB</th></tr>{scores}</table></div><p>문항별 실제 질문, 이미지, 정답과 답변 원문을 아래에 모두 제시합니다. 설정·소프트웨어·GPU·코드 기록은 manifest.json에 저장됩니다.</p></header><section><h2>관찰과 해석</h2><p>유형별 정답 수를 먼저 확인하고, 정오가 바뀐 문항의 이미지와 답변을 검토하세요. 출력 상한·형식 실패를 시각 인식 실패와 구분해야 합니다. 한 번의 대응 시드 비교만으로 일반적 개선을 결론 내릴 수 없습니다.</p><table><tr><th>유형</th>{''.join('<th>'+escape(n)+'</th>' for n in names)}</tr>{''.join(category_rows)}</table><p>정오 전환: {escape(json.dumps(summary.get('paired_transitions',{}),ensure_ascii=False))}</p></section>{''.join(cards)}</main></html>"
    (out / "index.html").write_text(html, encoding="utf-8")
    atomic_json(run / "summary.json", summary)
    atomic_json(run / "results.json", {"manifest": manifest, "summary": summary, "results": rows})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return out / "index.html"
