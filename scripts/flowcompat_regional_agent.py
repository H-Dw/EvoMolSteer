"""Independent regional Analyst/Designer profile; frozen v2 files stay unchanged."""
import argparse
import json

from evomolsteer.continuous.regional_workflow import (
    call_api, check_regional_receipt, compile_design, export_request,
    import_response, payload, verify_execution_response,
)
from evomolsteer.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("check", "export", "payload", "import", "api", "compile", "feedback"))
    parser.add_argument("--role", choices=("Analyst", "Designer"))
    for name in ("base-request", "regional-receipt", "reference", "request", "response", "analyst", "analyst-request", "program", "execution", "output"):
        parser.add_argument("--" + name)
    parser.add_argument("--round", type=int)
    parser.add_argument("--context", action="append", default=[], help="Bind an independent bounded textual audit; never claim it is a tool receipt")
    args = parser.parse_args()
    def require(*names):
        missing = [name for name in names if getattr(args, name) is None]
        if missing:
            parser.error("Required: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.action == "check":
        require("regional_receipt", "reference")
        value = check_regional_receipt(args.regional_receipt, args.reference)
        print(json.dumps({"passed": True, "reference_sha256": value["reference_sha256"], "receipt_sha256": value["receipt_sha256"], "coverage": value["manifest"]["coverage"]}))
    elif args.action == "export":
        require("base_request", "regional_receipt", "reference", "output", "role")
        path = export_request(args.base_request, args.regional_receipt, args.reference, args.output, args.role,
                              analyst=args.analyst, analyst_request=args.analyst_request, context_files=args.context)
        print(path)
    elif args.action == "payload":
        require("request")
        value = payload(read_json(args.request))
        from evomolsteer.io import digest
        value["request_sha256"] = digest(args.request)
        if args.output:
            write_json(args.output, value)
        else:
            print(json.dumps(value, ensure_ascii=True, allow_nan=False))
    elif args.action == "import":
        require("request", "response")
        import_response(args.request, args.response, args.output)
    elif args.action == "api":
        require("request")
        call_api(args.request)
    elif args.action == "compile":
        require("request", "response", "output", "round")
        compile_design(args.request, args.response, args.output, args.round)
    else:
        require("program", "execution", "output")
        verify_execution_response(args.program, args.execution, args.output)


if __name__ == "__main__":
    main()
