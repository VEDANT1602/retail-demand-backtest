"""Local experiment artifacts and integrity commitments; no network calls."""
import csv
import hashlib
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())

def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def model_write(path, model):
    with Path(path).open("wb") as file:
        pickle.dump(model, file, protocol=4)

def freeze(output):
    output = Path(output)
    sources = [ROOT / p for p in ["experiment.py", "experiment_io.py", "run.py"]]
    artifacts = sorted(p for p in output.iterdir() if p.is_file() and p.name != "manifest.json")
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources + artifacts}
    dump(output / "manifest.json", {"sha256": hashes})

def verify_manifest():
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
