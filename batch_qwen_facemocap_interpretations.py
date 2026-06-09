#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_qwen_facemocap_interpretations.py

Batch-generate deterministic FaceMoCap AE anatomical interpretations from
per-sample JSON files, then optionally ask a local Ollama model (e.g.,
qwen3:14b) to polish only the wording.

Python owns the facts. Qwen only polishes wording.

Example:
  python batch_qwen_facemocap_interpretations.py \
    --json_dir align_mean_movement_refactor/ae_region_side_llm_json_per_sample_VJ_OC/json \
    --out_dir align_mean_movement_refactor/ae_region_side_llm_interpretations_VJ_OC \
    --model qwen3:14b \
    --use_deterministic_on_fail
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

FORBIDDEN = [
    "diagnose", "diagnosis", "disease", "disorder", "neurolog", "clinical entity",
    "hemifacial", "dystonia", "spasm", "motor neuron", "facial nerve",
    "emg", "mri", "imaging", "botulinum", "toxin", "treatment", "specialist",
    "doctor", "healthcare", "medical evaluation", "further testing",
    "stroke", "tumor", "lesion", "neuropathy", "parkinson",
]

REQUIRED_LABELS = [
    "Interpretation:",
    "Movement context:",
    "Global burden:",
    "Metric pattern:",
    "Side pattern:",
    "Main anatomical findings:",
    "Caution:",
]

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: Any, max_len: int = 180) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:max_len].strip("_") or "sample"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_response(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    idx = text.find("Interpretation:")
    if idx >= 0:
        text = text[idx:]
    return text.strip()


def dominant_metrics(metric_burden: Dict[str, Dict[str, Any]], tolerance: int = 5) -> str:
    rows: List[Tuple[str, str, int, int]] = []
    for key, val in metric_burden.items():
        rows.append((
            key,
            val.get("display_name", key),
            int(val.get("class_3_to_4", 0)),
            int(val.get("class_4", 0)),
        ))
    if not rows:
        return "dominant metric cannot be determined"
    rows.sort(key=lambda x: (x[2], x[3]), reverse=True)
    first = rows[0]
    codom = [r for r in rows if abs(r[2] - first[2]) <= tolerance]
    if len(codom) > 1:
        return "co-dominant metrics are " + ", ".join(r[1] for r in codom)
    return f"dominant metric is {first[1]}"


def side_pattern(side_burden: Dict[str, Dict[str, Any]], tolerance: int = 5) -> str:
    left = side_burden.get("left", {})
    right = side_burden.get("right", {})
    center = side_burden.get("center", {})

    l34 = int(left.get("class_3_to_4", 0))
    r34 = int(right.get("class_3_to_4", 0))
    l4 = int(left.get("class_4", 0))
    r4 = int(right.get("class_4", 0))

    if abs(l34 - r34) <= tolerance:
        if r4 > l4:
            side_txt = "bilateral with slight right predominance"
        elif l4 > r4:
            side_txt = "bilateral with slight left predominance"
        else:
            side_txt = "bilateral without clear side predominance"
    elif r34 > l34:
        side_txt = "right-predominant"
    else:
        side_txt = "left-predominant"

    return (
        f"{side_txt}; left side has {l34} class 3-4 cells and {l4} class 4 cells, "
        f"right side has {r34} class 3-4 cells and {r4} class 4 cells, "
        f"and center regions have {int(center.get('class_3_to_4', 0))} class 3-4 cells "
        f"and {int(center.get('class_4', 0))} class 4 cells"
    )


def top_findings_text(top_findings: List[Dict[str, Any]], top_n: int = 5) -> str:
    top = top_findings[:top_n]
    if not top:
        return "no moderate-or-higher anatomical finding is available"
    return "; ".join(
        f"{x.get('metric_display', x.get('metric', 'metric'))} in the "
        f"{x.get('side', 'unknown side')} {x.get('region', 'unknown region')} region "
        f"(class {x.get('anomaly_class')}, {x.get('class_label')})"
        for x in top
    )


def build_deterministic_interpretation(data: Dict[str, Any], top_n: int = 5) -> str:
    meta = data.get("metadata", {})
    summary = data.get("summary_counts", {})
    metric = data.get("metric_burden", {})
    side = data.get("side_burden", {})
    top = data.get("top_findings", [])

    movement = meta.get("movement", "unknown movement")

    def c(key: str, default: int = 0) -> int:
        return int(summary.get(key, default))

    def m(metric_key: str, key: str = "class_3_to_4") -> int:
        return int(metric.get(metric_key, {}).get(key, 0))

    return (
        "Interpretation:\n"
        f"Movement context: The assessed movement is {movement}.\n"
        f"Global burden: The assessment contains {c('class_3_to_4')} high-or-outside-range "
        f"cells, including {c('class_4')} cells outside the healthy-evaluation range, "
        f"with {c('unavailable')} unavailable cells.\n"
        f"Metric pattern: The {dominant_metrics(metric)} based on class 3-4 burden; "
        f"trajectory abnormality has {m('trajectory_abnormality')} class 3-4 cells, "
        f"amplitude abnormality has {m('amplitude_abnormality')}, "
        f"hypokinesia has {m('hypokinesia')}, "
        f"hyperkinesia has {m('hyperkinesia')}, and "
        f"counter-direction ratio has {m('counter_direction_ratio')}.\n"
        f"Side pattern: The distribution is {side_pattern(side)}.\n"
        f"Main anatomical findings: The highest-ranked findings include {top_findings_text(top, top_n=top_n)}.\n"
        "Caution: These are AE-derived kinematic anomalies relative to healthy-evaluation percentiles, "
        "not a standalone clinical interpretation; unavailable or unreliable cells are not normal cells, "
        "and the result must be reviewed with the face-overlay PNG and target-vs-AE HTML animation."
    )


def build_polish_prompt(safe_text: str) -> str:
    return f"""
/no_think

Polish the following deterministic FaceMoCap interpretation as a concise research-style report.

STRICT RULES:
- Do not add any new facts.
- Do not paraphrase class labels; preserve phrases such as "outside healthy-evaluation range (>healthy max)" exactly when present.
- Do not change any numbers.
- Do not add disease names, diagnoses, treatments, exams, specialists, or clinical speculation.
- Do not add headings other than the existing labels.
- Keep the exact labels: Interpretation, Movement context, Global burden, Metric pattern, Side pattern, Main anatomical findings, Caution.
- If a phrase is awkward, improve wording only.

TEXT TO POLISH:
{safe_text}
"""


def call_ollama(
    prompt: str,
    model: str,
    url: str,
    temperature: float,
    top_p: float,
    num_ctx: int,
    num_predict: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response_obj:
            api_response = json.loads(response_obj.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {url}. Is 'ollama serve' running? Error: {exc}"
        ) from exc
    return clean_response(api_response.get("response", ""))


def check_response(response: str, forbidden_terms: List[str]) -> List[str]:
    issues: List[str] = []
    lower = response.lower()
    for term in forbidden_terms:
        if term.lower() in lower:
            issues.append(f"forbidden term: {term}")
    for label in REQUIRED_LABELS:
        if label not in response:
            issues.append(f"missing label: {label}")
    if "thinking" in lower or "okay, so" in lower or "let me" in lower:
        issues.append("contains thinking/preamble text")
    for heading in ["Key Findings", "Next Steps", "Clinical Context", "Recommendations", "Limitations", "Conclusion", "Differential"]:
        if heading.lower() in lower:
            issues.append(f"forbidden heading: {heading}")
    return issues


def find_json_files(json_dir: Path, pattern: str) -> List[Path]:
    files = sorted(json_dir.glob(pattern))
    return [
        p for p in files
        if not p.name.endswith("_summary.json") and p.name != "sample_json_summary.json"
    ]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="Directory containing per-sample JSON files")
    ap.add_argument("--out_dir", required=True, help="Output directory for interpretations")
    ap.add_argument("--pattern", default="*.json", help="Input JSON glob pattern")
    ap.add_argument("--model", default="qwen3:14b", help="Ollama model name")
    ap.add_argument("--ollama_url", default="http://localhost:11434/api/generate")
    ap.add_argument("--top_n_findings", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--num_predict", type=int, default=1000)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=None, help="Process only first N samples")
    ap.add_argument("--start_at", type=int, default=0, help="Skip the first N JSON files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing final outputs")
    ap.add_argument("--skip_qwen", action="store_true", help="Only generate deterministic interpretations")
    ap.add_argument("--use_deterministic_on_fail", action="store_true", help="Use deterministic text as final output when Qwen fails checks")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between Ollama calls")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    deterministic_dir = out_dir / "deterministic"
    polished_dir = out_dir / "polished"
    failed_dir = out_dir / "failed"
    final_dir = out_dir / "final"

    for d in [out_dir, deterministic_dir, final_dir]:
        ensure_dir(d)
    if not args.skip_qwen:
        for d in [polished_dir, failed_dir]:
            ensure_dir(d)

    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    json_files = find_json_files(json_dir, args.pattern)
    if args.start_at:
        json_files = json_files[args.start_at:]
    if args.limit is not None:
        json_files = json_files[:args.limit]
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir} with pattern {args.pattern}")

    index_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    print(f"[INFO] Samples to process: {len(json_files)}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Output: {out_dir}")

    for i, json_path in enumerate(json_files, start=1):
        sample_name = json_path.stem
        det_path = deterministic_dir / f"{sample_name}_deterministic_interpretation.txt"
        polished_path = polished_dir / f"{sample_name}_{safe_name(args.model)}_polished_interpretation.txt"
        failed_path = failed_dir / f"{sample_name}_{safe_name(args.model)}_polished_FAILED.txt"
        final_path = final_dir / f"{sample_name}_final_interpretation.txt"

        row = {
            "sample": sample_name,
            "json_path": str(json_path),
            "deterministic_path": str(det_path),
            "polished_path": str(polished_path) if not args.skip_qwen else "",
            "failed_path": "",
            "final_path": str(final_path),
            "status": "",
            "issues": "",
            "elapsed_sec": "",
        }
        sample_start = time.time()

        if final_path.exists() and not args.overwrite:
            row["status"] = "skipped_existing_final"
            row["elapsed_sec"] = round(time.time() - sample_start, 3)
            index_rows.append(row)
            print(f"[SKIP] {i}/{len(json_files)} {sample_name} final exists")
            continue

        try:
            data = load_json(json_path)
            safe_text = build_deterministic_interpretation(data, top_n=args.top_n_findings)
            det_path.write_text(safe_text + "\n", encoding="utf-8")

            if args.skip_qwen:
                final_path.write_text(safe_text + "\n", encoding="utf-8")
                row["status"] = "deterministic_only"
                print(f"[OK deterministic] {i}/{len(json_files)} {sample_name}")
            else:
                response = call_ollama(
                    prompt=build_polish_prompt(safe_text),
                    model=args.model,
                    url=args.ollama_url,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_ctx=args.num_ctx,
                    num_predict=args.num_predict,
                    timeout=args.timeout,
                )
                issues = check_response(response, forbidden_terms=FORBIDDEN)
                if issues:
                    failed_path.write_text(
                        response + "\n\n--- FAILED CHECK ---\n" + "\n".join(issues) + "\n",
                        encoding="utf-8",
                    )
                    row["failed_path"] = str(failed_path)
                    row["issues"] = " | ".join(issues)
                    if args.use_deterministic_on_fail:
                        final_path.write_text(safe_text + "\n", encoding="utf-8")
                        row["status"] = "qwen_failed_used_deterministic"
                        print(f"[WARN fallback] {i}/{len(json_files)} {sample_name}: {row['issues']}")
                    else:
                        row["status"] = "qwen_failed"
                        print(f"[FAIL] {i}/{len(json_files)} {sample_name}: {row['issues']}")
                else:
                    polished_path.write_text(response + "\n", encoding="utf-8")
                    final_path.write_text(response + "\n", encoding="utf-8")
                    row["status"] = "qwen_ok"
                    print(f"[OK qwen] {i}/{len(json_files)} {sample_name}")

            if args.sleep > 0:
                time.sleep(args.sleep)

        except Exception as exc:
            row["status"] = "error"
            row["issues"] = str(exc)
            ensure_dir(failed_dir)
            error_path = failed_dir / f"{sample_name}_ERROR.txt"
            error_path.write_text(str(exc) + "\n", encoding="utf-8")
            row["failed_path"] = str(error_path)
            print(f"[ERROR] {i}/{len(json_files)} {sample_name}: {exc}")

        row["elapsed_sec"] = round(time.time() - sample_start, 3)
        index_rows.append(row)
        pd.DataFrame(index_rows).to_csv(out_dir / "batch_interpretation_index.csv", index=False)

    elapsed = time.time() - t0
    index_df = pd.DataFrame(index_rows)
    index_path = out_dir / "batch_interpretation_index.csv"
    summary_path = out_dir / "batch_interpretation_summary.json"
    index_df.to_csv(index_path, index=False)
    summary = {
        "json_dir": str(json_dir),
        "out_dir": str(out_dir),
        "model": args.model,
        "n_samples": int(len(index_df)),
        "status_counts": index_df["status"].value_counts(dropna=False).to_dict() if not index_df.empty else {},
        "elapsed_sec": round(elapsed, 3),
        "skip_qwen": bool(args.skip_qwen),
        "use_deterministic_on_fail": bool(args.use_deterministic_on_fail),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE]")
    print(f"  Index: {index_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Final interpretations: {final_dir}")
    print(f"  Status counts: {summary['status_counts']}")
    print(f"  Elapsed seconds: {summary['elapsed_sec']}")


if __name__ == "__main__":
    main()
