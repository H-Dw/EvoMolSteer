"""Command-line dataset boundary for verified trajectory conversion."""
import argparse
import json
from pathlib import Path

from .lifecycle import convert_dataset, convert_trajectory


def main(argv=None):
    parser = argparse.ArgumentParser(description="Losslessly optimize a closed NPZ or completed generation dataset")
    parser.add_argument("--input", required=True, help="NPZ file or generation dataset root")
    parser.add_argument("--output", required=True, help="HDF5 file or output dataset root")
    parser.add_argument("--delete-source", action="store_true",
                        help="Retire only verified NPZ inputs after successful conversion")
    args = parser.parse_args(argv)
    source = Path(args.input)
    if source.suffix == ".npz":
        result = convert_trajectory(source, args.output, input_dataset=source.resolve().parent,
                                    delete_source=args.delete_source)
    else:
        result = convert_dataset(source, args.output, delete_source=args.delete_source)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
