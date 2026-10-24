"""
pipeline/analysis/_output.py — the one output convention (Principle 8).

save(name, results, render) writes
  results/analysis/<name>.json   {"meta": provenance, "results": numbers at full precision}
  results/analysis/<name>.md     the readable report, rendered from the JSON as written
and prints the report. Rendering from the written JSON (not from in-memory values) means the report can never
show a number the JSON lacks. Figures, paper tables and claim checks read the JSON.

Conventions for `results`: proportions and differences stay fractions; each section carries "unit" and "n";
estimates are {"est", "lo", "hi", "se"}; keys use config ids, display names come from meta.study.names.
"""
import contextlib
import datetime
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"


def jsonable(x):
    """numpy scalars and arrays, tuples and non-string keys -> plain JSON types."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def meta(name):
    spec = getattr(sys.modules["__main__"], "__spec__", None)
    entry = f"python -m {spec.name}" if spec else f"python {sys.argv[0]}"
    study = yaml.safe_load((ROOT / "config/default.yaml").read_text()).get("study", {})
    return {"analysis": name, "command": " ".join([entry, *sys.argv[1:]]),
            "commit": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--", "pipeline", "scripts", "config")),
            "date": datetime.date.today().isoformat(), "study": study}


def save(name, results, render=None, fenced=True):
    """Write <name>.json and, given render (results dict -> prints the report), <name>.md.
    fenced: the report is fixed-width text, wrapped in a code block; False if render prints Markdown."""
    text = json.dumps({"meta": meta(name), "results": jsonable(results)}, indent=1, ensure_ascii=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(text + "\n")
    if render is None:
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        render(json.loads(text)["results"])
    report = buf.getvalue()
    sys.stdout.write(report)
    head = f"<!-- Rendered from {name}.json; regenerate, do not edit. -->\n\n"
    (OUT / f"{name}.md").write_text(head + (f"```text\n{report.rstrip()}\n```\n" if fenced else report))


def load(name):
    """What figure scripts and table generators call: the results block of <name>.json."""
    return json.loads((OUT / f"{name}.json").read_text())["results"]
