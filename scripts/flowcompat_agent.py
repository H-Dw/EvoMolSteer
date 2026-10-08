"""Bound Analyst/Designer calls, deterministic tool receipts and response gate."""
import argparse
from evomolsteer.continuous.flowcompat_agents import (
    audit_flow_configuration, call_api, compile_design, export_request,
    import_response, run_registered_tool, verify_execution_response,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", required=True, choices=("tool", "audit-flow", "export", "import", "api", "compile", "feedback"))
    p.add_argument("--role", choices=("Analyst", "Designer"))
    p.add_argument("--evidence"); p.add_argument("--registry")
    p.add_argument("--receipt", action="append", default=[])
    p.add_argument("--plan"); p.add_argument("--request"); p.add_argument("--response")
    p.add_argument("--analyst"); p.add_argument("--program"); p.add_argument("--reference")
    p.add_argument("--source-module", action="append", default=[])
    p.add_argument("--model-audit"); p.add_argument("--execution")
    p.add_argument("--output"); p.add_argument("--round", type=int)
    a = p.parse_args()
    def require(*names):
        missing = [name for name in names if not getattr(a, name)]
        if missing: p.error("Required for this action: " + ", ".join("--" + name for name in missing))
    if a.action == "tool":
        require("plan", "output"); run_registered_tool(a.plan, a.output)
    elif a.action == "audit-flow":
        require("program", "reference", "output")
        audit_flow_configuration(a.program, a.reference, a.output, a.source_module, a.model_audit)
    elif a.action == "export":
        require("evidence", "registry", "output", "role", "receipt")
        export_request(a.evidence, a.registry, a.receipt, a.output, a.role, analyst=a.analyst)
    elif a.action == "import":
        require("request", "response"); import_response(a.request, a.response, a.output)
    elif a.action == "api":
        require("request"); call_api(a.request)
    elif a.action == "compile":
        require("request", "response", "output", "round")
        compile_design(a.request, a.response, a.output, a.round)
    else:
        require("program", "execution", "output")
        verify_execution_response(a.program, a.execution, a.output)


if __name__ == "__main__":
    main()
