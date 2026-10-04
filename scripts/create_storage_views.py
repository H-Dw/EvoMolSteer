import argparse
from evomolsteer.storage.views import build_shard_views

p = argparse.ArgumentParser(description="Verify redundant shards and create shared-package views")
p.add_argument("--packages", required=True)
p.add_argument("--shard-root", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
build_shard_views(a.packages, a.shard_root, a.output)
