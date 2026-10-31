"""Use the existing audited, plan-driven read-only inspector for this new suite."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "20261031_guidance_comparison5/inspect_suite.py"), run_name="__main__")
