"""Analyze all observed selection nodes without fixed time strata."""
import argparse

from evomolsteer.temporal_guidance.mining import mine_profiles


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moments", required=True, help="Compact batch_node_moments.parquet")
    p.add_argument("--prior", required=True, help="Feature scales, times and window JSON")
    p.add_argument("--packet", required=True, help="Frozen evidence clocks/label semantics JSON")
    p.add_argument("--output", required=True, help="stage_profiles.json; no expanded feature cache")
    p.add_argument("--max-knots", type=int, default=2)
    p.add_argument("--minimum-nodes", type=int, default=6)
    a = p.parse_args()
    mine_profiles(a.moments, a.prior, a.packet, a.output, max_knots=a.max_knots, minimum_nodes=a.minimum_nodes)
