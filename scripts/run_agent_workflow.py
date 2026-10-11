"""Standard file-in/file-out API entrypoint; no molecular inference dispatched."""
import argparse
from pathlib import Path

from evomolsteer.agent_workflow import EvidenceTools, run_agent
from evomolsteer.io import read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["Analyst", "Designer"], required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--choices", type=Path)
    parser.add_argument("--handoff-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    tools = EvidenceTools(args.evidence_dir)
    handoff = None
    if args.handoff_dir:
        tools.import_references(read_json(args.handoff_dir / "references.json"))
        handoff = read_json(args.handoff_dir / "Analyst.json")
    repo = Path(__file__).resolve().parents[1]
    skill = repo / "skills/agent-json-interface/SKILL.md"
    instructions = args.instructions.read_text(encoding="utf-8") + "\n" + skill.read_text(encoding="utf-8")
    instructions += "\n" + (skill.parent / "references/wire_examples.md").read_text(encoding="utf-8")
    run_agent(args.role, tools, instructions, read_json(args.task), args.output_dir,
              choices=read_json(args.choices) if args.choices else None, handoff=handoff,
              max_steps=args.max_steps)
    print("Validated, source-bound output:", args.output_dir / (args.role + ".json"))


if __name__ == "__main__":
    main()
