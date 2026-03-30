import json
from pathlib import Path

def clean_path(path):
    if not path:
        return []
    path_tuples = [tuple(p) if isinstance(p, list) else (p.get("x"), p.get("y")) for p in path]
    cleaned = []
    for coord in path_tuples:
        if coord in cleaned:
            idx = cleaned.index(coord)
            cleaned = cleaned[:idx]
        cleaned.append(coord)
    return cleaned

results_path = Path("results/spatial_gym")
anomalies = []
for jsonl_file in results_path.glob("*_gym_traceback.jsonl"):
    with open(jsonl_file) as f:
        for line in f:
            entry = json.loads(line)
            result = entry.get("result", {})
            steps = result.get("steps_taken", 0)
            path = result.get("extracted_path", [])
            cleaned = clean_path(path)
            if steps > 0 and len(cleaned) > steps:
                anomalies.append((steps, len(path), len(cleaned), path[:15]))

print(f"Found {len(anomalies)} anomalies where cleaned_path > steps")
if anomalies:
    print("First 5 examples (steps, raw_len, cleaned_len):")
    for steps, raw, clean, path in anomalies[:5]:
        print(f"  steps={steps}, raw={raw}, cleaned={clean}")
        print(f"    path={path}")
