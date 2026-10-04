"""Verify and restore an archive created by archive_generation.py."""
import argparse
import json

from evomolsteer.io import write_json
from evomolsteer.storage.generation_archive import verify_generation_archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = verify_generation_archive(args.archive, args.metadata, args.destination)
    write_json(args.report, result)
    print(json.dumps(result, indent=2))
