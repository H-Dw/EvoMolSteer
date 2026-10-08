"""Reference moments to compact positive/negative temporal target functions."""
import argparse
import json
from evomolsteer.continuous.cohort_target_functions import fit_targets


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--maximum-degree', type=int, default=3)
    a = p.parse_args()
    result = fit_targets(a.reference, a.output, a.maximum_degree)
    print(json.dumps({name: {f: v['degree'] for f, v in values.items()}
                      for name, values in result['functions'].items()}))
