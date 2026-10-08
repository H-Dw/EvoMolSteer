"""Versioned third-tool role workflow. Existing flowcompat_agent.py is unchanged.

Migration: tool --plan NEW_BRANCH_PLAN -> export with three --receipt options,
--evidence OLD_TOOL_EVIDENCE and --branch-evidence NEW_BRANCH_EVIDENCE; import or
api -> compile. Requests/responses use distinct flowcompat-supplemental filenames.
"""
import argparse
from evomolsteer.continuous.flowcompat_supplemental import (
    call_api, compile_design, export_request, import_response, run_branch_tool,
    verify_execution_response,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", required=True, choices=("tool", "export", "import", "api", "compile", "feedback"))
    p.add_argument("--role", choices=("Analyst", "Designer"))
    p.add_argument("--evidence"); p.add_argument("--branch-evidence"); p.add_argument("--registry")
    p.add_argument("--extra-evidence"); p.add_argument("--extra-receipt")
    p.add_argument("--receipt", action="append", default=[])
    p.add_argument("--plan"); p.add_argument("--request"); p.add_argument("--response"); p.add_argument("--analyst")
    p.add_argument("--program"); p.add_argument("--execution"); p.add_argument("--output"); p.add_argument("--round", type=int)
    a = p.parse_args()
    def require(*fields):
        missing = [v for v in fields if not getattr(a, v)]
        if missing: p.error("Required: " + ", ".join("--" + v.replace("_", "-") for v in missing))
    if a.action == "tool":
        require("plan", "output"); run_branch_tool(a.plan, a.output)
    elif a.action == "export":
        require("evidence", "branch_evidence", "registry", "receipt", "output", "role")
        export_request(a.evidence, a.branch_evidence, a.registry, a.receipt, a.output, a.role, analyst=a.analyst,
                       extra_evidence=a.extra_evidence, extra_receipt=a.extra_receipt)
    elif a.action == "import":
        require("request", "response"); import_response(a.request, a.response, a.output)
    elif a.action == "api":
        require("request"); call_api(a.request)
    elif a.action == "compile":
        require("request", "response", "output", "round"); compile_design(a.request, a.response, a.output, a.round)
    else:
        require("program", "execution", "output"); verify_execution_response(a.program, a.execution, a.output)


if __name__ == "__main__": main()
