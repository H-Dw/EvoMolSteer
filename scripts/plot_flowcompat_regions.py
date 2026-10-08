"""Export frozen regional covariance evidence as figures and a compact table.

No new significance testing or inference: the input is an attested synthesis
dataset. Displayed directions are associations, not learned affinity gradients.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evomolsteer.io import digest, read_json, write_json


def plot(input_dataset, output_dataset):
    source, output = Path(input_dataset).resolve(), Path(output_dataset).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError('Keep figure output outside the attested input dataset')
    rules_path = source / 'regional_rules.json'
    rules = read_json(rules_path)
    regions = rules['selected_regions']
    if not regions:
        raise ValueError('There are no qualified regions to plot')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for region in regions:
        for axis, name in enumerate('xyz'):
            rows.append({'region': region['region'], 'ancestry_depth': region['depth'],
                         'component': name, 'covariance_A_pic50': region['whole_window_xyz'][axis],
                         'CI_low': region['component_CI_low'][axis],
                         'CI_high': region['component_CI_high'][axis],
                         'q': region['component_q'][axis],
                         'fitted_degree': region['functions_xyz'][name]['degree'],
                         'landmark_x_A': region['landmark_A'][0],
                         'landmark_y_A': region['landmark_A'][1],
                         'landmark_z_A': region['landmark_A'][2],
                         'independent_batches': region['independent_batches'],
                         'LOO_positive_fraction': region['loo_positive_fraction']})
    pd.DataFrame(rows).to_csv(output / 'regional_covariance_evidence.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, max(4, len(regions) * .55)))
    positions = np.arange(len(regions))
    for axis, (name, color) in enumerate(zip('xyz', ['#2368a0', '#c2502f', '#398453'])):
        mean = np.array([r['whole_window_xyz'][axis] for r in regions]) * 1e6
        low = np.array([r['component_CI_low'][axis] for r in regions]) * 1e6
        high = np.array([r['component_CI_high'][axis] for r in regions]) * 1e6
        y = positions + (axis - 1) * .22
        ax.hlines(y, low, high, color=color, linewidth=1.3)
        significant = np.array([r['component_q'][axis] < rules['q_threshold']
                                and r['component_CI_low'][axis] * r['component_CI_high'][axis] > 0
                                for r in regions])
        ax.scatter(mean[significant], y[significant], color=color, s=24, label=name)
        ax.scatter(mean[~significant], y[~significant], edgecolor=color,
                   facecolor='white', linewidth=1.3, s=24)
    ax.axvline(0, color='#666666', linewidth=.8)
    ax.set_yticks(positions, [f'ROI {r["region"]:02d}' for r in regions])
    ax.set_xlabel('Whole-window adjusted covariance (10^-6 A * pIC50)')
    ax.set_title(f'Observed regional XYZ associations; common ancestry depth {rules["common_depth"]}')
    ax.legend(title='Coordinate', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.12, .015, 'Filled: supplied q/CI qualification. Hollow: other components retained in the joint field.', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(output / 'regional_covariance.png', dpi=180)
    plt.close(fig)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    display_points = []
    for region in regions:
        center = np.array(region['landmark_A'], float)
        vector = np.array(region['whole_window_xyz'], float)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError('Qualified regional vector cannot be zero')
        display_points.extend((center, center + 1.5 * vector / norm))
        ax.scatter(*center, color='#2368a0', s=30)
        ax.quiver(*center, *(vector / norm), length=1.5, color='#c2502f', arrow_length_ratio=.25)
        ax.text(*center, f' {region["region"]:02d}', fontsize=9)
    ax.set(xlabel='World x (A)', ylabel='World y (A)', zlabel='World z (A)',
           title='Regional landmarks and normalized association directions')
    # Equal box dimensions alone distort world-space directions when axis
    # ranges differ. Include arrow tips, then use the same range on all axes.
    points = np.asarray(display_points)
    low, high = points.min(axis=0), points.max(axis=0)
    midpoint = (low + high) / 2
    half_range = max(float((high - low).max()) / 2, .5) + .5
    for axis, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(midpoint[axis] - half_range, midpoint[axis] + half_range)
    ax.set_box_aspect((1, 1, 1))
    fig.text(.09, .03, 'Arrows have equal display length; they are not physical displacement or causal force.', fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(output / 'regional_landmarks.png', dpi=180)
    plt.close(fig)
    write_json(output / 'manifest.json', {
        'input_dataset': str(source), 'input_files': {str(rules_path): digest(rules_path)},
        'script_sha256': digest(__file__), 'window': rules['window'],
        'common_depth': rules['common_depth'],
        'interpretation': 'Frozen observed associations; no new tests, biological validation or attention attribution',
        'outputs': {p.name: digest(p) for p in sorted(output.iterdir())
                    if p.is_file() and p.name != 'manifest.json'},
    })
    return {'regions': len(regions), 'components': len(rows)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset', required=True)
    parser.add_argument('--output-dataset', required=True)
    args = parser.parse_args()
    print(plot(args.input_dataset, args.output_dataset))
