"""Lightweight checks that do not require the private or published SCAPS dataset."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_all_stage_scripts_compile():
    for path in sorted((ROOT / "stages").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ast.parse((ROOT / "run_pipeline.py").read_text(encoding="utf-8"))

def test_help_cli():
    completed = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--help"],
                               capture_output=True, text=True, check=True)
    assert "--dataset" in completed.stdout
    assert "--quick-check" in completed.stdout

def test_original_notebook_is_present():
    assert (ROOT / "notebooks" / "SCAPS_ML_original_source.ipynb").is_file()
