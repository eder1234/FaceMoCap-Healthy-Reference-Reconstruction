
import json
import re
import urllib.request
import urllib.error
from pathlib import Path

MODEL = "qwen3:14b"
OLLAMA_URL = "http://localhost:11434/api/generate"

JSON_PATH = Path(
    "align_mean_movement_refactor/ae_region_side_llm_json_per_sample_VJ_OC/"
    "json/anonymous_sample_0001.json"
)

OUT_SAFE = JSON_PATH.with_suffix(".deterministic_interpretation.txt")
OUT_POLISHED = JSON_PATH.with_suffix(".qwen3_14b_polished_interpretation.txt")
OUT_FAILED = JSON_PATH.with_suffix(".qwen3_14b_polished_FAILED.txt")

FORBIDDEN = [
    "diagnose", "diagnosis", "disease", "disorder", "neurolog", "clinical entity",
    "hemifacial", "dystonia", "spasm", "motor neuron", "facial nerve",
    "emg", "mri", "imaging", "botulinum", "toxin", "treatment", "specialist",
    "doctor", "healthcare", "medical evaluation", "further testing"
]

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

meta = data["metadata"]
summary = data["summary_counts"]
metric = data["metric_burden"]
side = data["side_burden"]
top = data["top_findings"][:5]

def dominant_metrics(metric_burden):
    rows = []
    for key, val in metric_burden.items():
        rows.append((key, val["display_name"], val["class_3_to_4"], val["class_4"]))
    rows.sort(key=lambda x: (x[2], x[3]), reverse=True)
    first = rows[0]
    codom = [r for r in rows if abs(r[2] - first[2]) <= 5]
    if len(codom) > 1:
        names = ", ".join(r[1] for r in codom)
        return f"co-dominant metrics are {names}"
    return f"dominant metric is {first[1]}"

def side_pattern(side_burden):
    left = side_burden.get("left", {})
    right = side_burden.get("right", {})
    center = side_burden.get("center", {})

    l34, r34 = left.get("class_3_to_4", 0), right.get("class_3_to_4", 0)
    l4, r4 = left.get("class_4", 0), right.get("class_4", 0)

    if abs(l34 - r34) <= 5:
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
        f"and center regions have {center.get('class_3_to_4', 0)} class 3-4 cells "
        f"and {center.get('class_4', 0)} class 4 cells"
    )

top_txt = "; ".join(
    f"{x['metric_display']} in the {x['side']} {x['region']} region "
    f"(class {x['anomaly_class']}, {x['class_label']})"
    for x in top
)

safe_text = (
    "Interpretation:\n"
    f"Movement context: The assessed movement is {meta.get('movement')}.\n"
    f"Global burden: The assessment contains {summary['class_3_to_4']} high-or-outside-range "
    f"cells, including {summary['class_4']} cells outside the healthy-evaluation range, "
    f"with {summary['unavailable']} unavailable cells.\n"
    f"Metric pattern: The {dominant_metrics(metric)} based on class 3-4 burden; "
    f"trajectory abnormality has {metric['trajectory_abnormality']['class_3_to_4']} class 3-4 cells, "
    f"amplitude abnormality has {metric['amplitude_abnormality']['class_3_to_4']}, "
    f"hypokinesia has {metric['hypokinesia']['class_3_to_4']}, "
    f"hyperkinesia has {metric['hyperkinesia']['class_3_to_4']}, and "
    f"counter-direction ratio has {metric['counter_direction_ratio']['class_3_to_4']}.\n"
    f"Side pattern: The distribution is {side_pattern(side)}.\n"
    f"Main anatomical findings: The highest-ranked findings include {top_txt}.\n"
    "Caution: These are AE-derived kinematic anomalies relative to healthy-evaluation percentiles, "
    "not a standalone clinical interpretation; unavailable or unreliable cells are not normal cells, "
    "and the result must be reviewed with the face-overlay PNG and target-vs-AE HTML animation."
)

OUT_SAFE.write_text(safe_text + "\n", encoding="utf-8")

prompt = f"""
/no_think

Polish the following deterministic FaceMoCap interpretation as a concise research-style report.

STRICT RULES:
- Do not add any new facts.
- Do not paraphrase class labels; preserve phrases such as "outside healthy-evaluation range (>healthy max)" exactly.
- Do not change any numbers.
- Do not add disease names, diagnoses, treatments, exams, specialists, or clinical speculation.
- Do not add headings other than the existing labels.
- Keep the exact labels: Interpretation, Movement context, Global burden, Metric pattern, Side pattern, Main anatomical findings, Caution.
- If a phrase is awkward, improve wording only.

TEXT TO POLISH:
{safe_text}
"""

payload = {
    "model": MODEL,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.1,
        "top_p": 0.8,
        "num_ctx": 4096,
        "num_predict": 1000
    }
}

request = urllib.request.Request(
    OLLAMA_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(request, timeout=900) as response_obj:
        api_response = json.loads(response_obj.read().decode("utf-8"))
except urllib.error.URLError as e:
    raise SystemExit(f"Could not connect to Ollama at {OLLAMA_URL}. Is 'ollama serve' running? Error: {e}")

response = api_response.get("response", "").strip()

# Defensive cleanup: remove ANSI terminal escape sequences if any appear.
response = ANSI_RE.sub("", response)

# Remove thinking or any preamble if Qwen still emits it.
idx = response.find("Interpretation:")
if idx >= 0:
    response = response[idx:].strip()

issues = []
lower = response.lower()

for w in FORBIDDEN:
    if w in lower:
        issues.append(f"forbidden term: {w}")

for label in [
    "Interpretation:",
    "Movement context:",
    "Global burden:",
    "Metric pattern:",
    "Side pattern:",
    "Main anatomical findings:",
    "Caution:",
]:
    if label not in response:
        issues.append(f"missing label: {label}")

if issues:
    OUT_FAILED.write_text(
        response + "\n\n--- FAILED CHECK ---\n" + "\n".join(issues),
        encoding="utf-8"
    )
    print(response)
    print("\n[FAILED CHECK]")
    for issue in issues:
        print("-", issue)
    print(f"\nSaved failed response to: {OUT_FAILED}")
else:
    OUT_POLISHED.write_text(response + "\n", encoding="utf-8")
    print(response)
    print(f"\n[OK] Deterministic draft: {OUT_SAFE}")
    print(f"[OK] Polished response: {OUT_POLISHED}")

