"""Run the original experiment on CPU; alternate output paths preserve reference artifacts."""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
from experiment import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    root = Path(__file__).resolve().parent
    if not output.is_relative_to(root):
        parser.error("output must remain within this repository")
    output.mkdir(parents=True, exist_ok=True)
    run(output)
