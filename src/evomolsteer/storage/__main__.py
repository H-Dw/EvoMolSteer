"""Run with python -m evomolsteer.storage COMMAND ..."""
import argparse
import json

from .features import pack_features, restore_features, verify_features
from .trajectory import TrajectoryPackage, pack_trajectory


def main():
    parser = argparse.ArgumentParser(description="Lossless node packages; trajectory gzip + shuffle is optional")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["pack-trajectory", "pack-features", "restore-trajectory", "restore-features"]:
        p = commands.add_parser(name)
        p.add_argument("source")
        p.add_argument("destination")
        if name == "pack-trajectory":
            p.add_argument("--dense", action="store_true")
            p.add_argument('--codec',choices=['none','gzip_shuffle'],default='none')
        if name == "pack-features":
            p.add_argument("--limit-partitions", type=int)
    for name in ["verify-trajectory", "verify-features"]:
        p = commands.add_parser(name)
        p.add_argument("source")
        if name == "verify-features":
            p.add_argument("--original")
    a = parser.parse_args()
    if a.command == "pack-trajectory":
        result = pack_trajectory(a.source, a.destination, normalize=not a.dense,codec=a.codec)
    elif a.command == "pack-features":
        result = pack_features(a.source, a.destination, a.limit_partitions)
    elif a.command == "restore-features":
        result = {"restored": str(restore_features(a.source, a.destination))}
    elif a.command == "verify-features":
        result = verify_features(a.source, a.original)
    else:
        with TrajectoryPackage(a.source) as package:
            if a.command == "restore-trajectory":
                package.restore(a.destination)
                result = {"restored": a.destination}
            else:
                result = {"verified_arrays": package.verify()}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
