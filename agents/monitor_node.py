import time
import json
from pathlib import Path

WATCH_PATH = Path("data/incoming")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

def run(params, context):
    items = []
    for p in WATCH_PATH.glob("*"):
        if p.is_file():
            items.append({"path": str(p), "name": p.name})
    out = {"files": items}
    out_path = PROCESSED / "monitor_output.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out

if __name__ == "__main__":
    print(run({}, {}))