"""Build compact, independently typed native/copy lineage edges."""
import argparse
from evomolsteer.continuous.path_graph import build_path_graph

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "campaign", "output"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--window", type=float, nargs=2, required=True)
    p.add_argument("--arm", default="single")
    p.add_argument("--batches", type=lambda s: [int(v) for v in s.split(",")])
    a=p.parse_args()
    m=build_path_graph(a.dataset,a.campaign,a.output,a.window,a.arm,a.batches)
    print({k:m[k] for k in ("schema_version","window","nodes","edges","tables")})
