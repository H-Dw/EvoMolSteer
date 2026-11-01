"""Run the existing audited sequential campaign with this suite's frozen PLAN."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    common=Path(__file__).resolve().parents[1]/"20261031_guidance_comparison5"
    sys.path.insert(0,str(common))
    runpy.run_path(str(common/"run_suite.py"),run_name="__main__")
