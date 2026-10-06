"""Compile a compact directional reference from complete discovery-window data."""
import argparse
from evomolsteer.generation.coordinate_shape import build

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('dataset', 'campaign', 'mining', 'output', 'regions'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--channel', choices=['all', 'NOS'], default='NOS')
    a = p.parse_args()
    reference = build(a.dataset, a.campaign, a.mining, a.output, a.regions.split(','), a.channel)
    print({'window': reference['window'], 'n_modes': sum(len(f['modes']) for f in reference['frames']),
           'features': reference['features'], 'scales_A2': reference['feature_scale_A2']})
