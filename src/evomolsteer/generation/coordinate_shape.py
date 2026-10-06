"""Discovery-only directional shape targets, in explicitly dimensionless units.

This is a representation ablation, not evidence of a causal affinity mechanism.
No neural model is fitted and no per-particle feature cache is written.
"""
import gzip
import json
import math
from pathlib import Path
import numpy as np
import torch
from ..continuous.coordinate_features import regional_observables
from ..io import read_json, digest
from ..storage.trajectory import TrajectoryPackage
from .coordinate_reward import CoordinateMixtureReward
from .prototypes import write_json

COMPONENTS = ('xx', 'yy', 'zz', 'xy', 'xz', 'yz')


def time_weights(times):
    """Normalized trapezoidal quadrature over the complete observed window."""
    times = np.asarray(times, float)
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError('Increasing complete-window times required')
    w = np.empty(len(times))
    w[0] = (times[1]-times[0])/2
    w[-1] = (times[-1]-times[-2])/2
    w[1:-1] = (times[2:]-times[:-2])/2
    return w/w.sum()


def dimensionless_moments(v, weights, scale):
    """Fixed whole-window scales; covariance shrinkage has no angstrom floor."""
    z = np.asarray(v, float)/scale
    weights = np.asarray(weights, float)
    weights = weights/weights.sum()
    center = weights@z
    d = z-center
    cov = np.einsum('bi,b,bj->ij', d, weights, d)
    covariance = .9*cov + .1*np.diag(np.diag(cov)) + .01*np.eye(z.shape[1])
    return {'center_scaled': center.tolist(), 'covariance_dimensionless': covariance.tolist(),
            'center_A2': (center*scale).tolist(), 'raw_covariance_A4': (cov*scale[:, None]*scale[None, :]).tolist()}


def build(dataset, campaign, mining, output, regions, channel='NOS'):
    root, mining, output = map(Path, (dataset, mining, output))
    if output.exists():
        raise FileExistsError(output)
    manifest, catalog = read_json(mining/'manifest.json'), read_json(mining/'feature_catalog.json')
    if manifest.get('feature_family') != 'shape' or manifest.get('spatial_anchor') != 'endpoint' or manifest.get('control_representation') != 'proposal':
        raise ValueError('Endpoint-anchored proposal shape evidence required')
    if not regions or len(set(regions)) != len(regions) or set(regions)-set(catalog['regions']) or channel not in ('all', 'NOS'):
        raise ValueError('Explicit measured regions and channel required')
    regions = sorted(regions)
    subcatalog = {**catalog, 'regions': {r: catalog['regions'][r] for r in regions}}
    features = [f'{r}::{channel}::proposal_shape_{k}' for r in regions for k in COMPONENTS]
    cfg = read_json(root/'results'/campaign/'config.json')
    batches = manifest['splits']['discovery']
    frames, sources = {}, []
    for batch in batches:
        path = root/'results'/campaign/'single'/f'batch_{batch:03d}'/'trajectory.h5'
        com = np.asarray(read_json(root/'results'/campaign/f'frame_batch_{batch:03d}.json')['target_com'])[:, None]
        with TrajectoryPackage(path) as package:
            times = np.round(package.read('score_time')[:, 0].astype(float), 6)
            state_times = package.read('state_time').astype(float)
            for i in np.flatnonzero(package.read('resampled')):
                t = float(times[i])
                if t not in manifest['times']:
                    continue
                x, end, proposal = [package.read(f'{view}_coords', int(i)).astype(float)*cfg['coord_scale']+com
                                    for view in ('current', 'predicted', 'proposal')]
                atoms, mask = package.read('predicted_atomics', int(i)), package.read('mask', int(i))
                values, names, _ = regional_observables(x, end, proposal, atoms, mask, subcatalog, t, float(state_times[i])-t,
                    catalog['spatial_width_A'], 'endpoint', True, 'shape')
                v = values[:, [names.index(f) for f in features]]
                eligible = mask.astype(bool)
                if channel == 'NOS':
                    eligible &= np.isin(atoms, [catalog['atom_vocabulary'][a] for a in ('N', 'O', 'S')])
                valid = np.isfinite(v).all(1) & (eligible.sum(1) >= 2)
                weight = package.read('selection_probability', int(i)).astype(float)[valid]
                if valid.sum() < 3 or weight.sum() <= 0:
                    raise ValueError('Insufficient informative candidate support')
                frames.setdefault(t, []).append((batch, v[valid], weight/weight.sum(), float(valid.mean())))
        sources.append({'path': path.relative_to(root).as_posix(), 'sha256': digest(path)})
    times = sorted(frames)
    if times != manifest['times'] or any([b for b, *_ in frames[t]] != batches for t in times):
        raise ValueError('Incomplete discovery batch/time grid')
    expected = [s for s in manifest['sources'] if Path(s['path']).parent.parent.name == 'single'
                and int(Path(s['path']).parent.name.split('_')[-1]) in batches]
    if sources != expected:
        raise ValueError('Shape evidence/source trajectory hashes differ')
    variance = np.asarray([np.mean([v.var(0) for _, v, _, _ in frames[t]], axis=0) for t in times])
    scale = np.sqrt(time_weights(times)@variance)
    if not np.isfinite(scale).all() or (scale <= 1e-12).any():
        raise ValueError('Unestimable whole-window feature scale; defer this design')
    result_frames = []
    for t in times:
        modes = [{**dimensionless_moments(v, w, scale), 'source_batch': b, 'availability': a,
                  'n_available': len(v), 'selected_probability_ESS': float(1/(w@w))}
                 for b, v, w, a in frames[t]]
        result_frames.append({'time': t, 'modes': modes})
    from .launcher import INPUT_FILES
    ref = {'schema_version': 'regional-shape-mixture-1.0', 'window': manifest['window'], 'times': times,
        'regions': subcatalog['regions'], 'channel': channel, 'atom_vocabulary': catalog['atom_vocabulary'],
        'spatial_width_A': catalog['spatial_width_A'], 'spatial_anchor': 'endpoint', 'control_representation': 'proposal',
        'time_alignment': 'score time t -> proposal state time t+dt; fixed endpoint anchor at t',
        'representation': 'regional weighted central second moment in aligned receptor frame; svec off diagonals sqrt(2)',
        'features': features, 'feature_unit': 'A^2', 'feature_scale_A2': scale.tolist(),
        'scale_definition': 'Trapezoid whole-window mean of equal-batch unweighted within-candidate population variance; square root. Fixed across all time nodes.',
        'covariance_definition': '.9*C + .1*diag(C) + .01*I in dimensionless scaled feature space',
        'frames': result_frames, 'batches': batches, 'sources': sources,
        'required_input_sha256': {name: digest(root/'inputs'/name) for name in INPUT_FILES},
        'source_manifest_sha256': digest(mining/'manifest.json'),
        'limitations': ['Exploratory representation ablation, not a discovered causal affinity direction',
                        'Six tensor components contain trace; they are not pure anisotropy',
                        'Eligible>=2 does not imply effective regional support>=2',
                        'Central moments do not uniquely determine molecular geometry or bonds']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress((json.dumps(ref, separators=(',', ':'), allow_nan=False)+'\n').encode(), mtime=0))
    write_json(output.with_suffix('.manifest.json'), {'sha256': digest(output), 'bytes': output.stat().st_size,
        'window': ref['window'], 'regions': regions, 'channel': channel, 'feature_scale_A2': scale.tolist(),
        'source_manifest_sha256': ref['source_manifest_sha256']})
    return ref


class CoordinateShapeReward(CoordinateMixtureReward):
    def __init__(self, program, reference):
        self.program, self.reference = program, reference
        self.window, self.times = reference['window'], np.asarray(reference['times'])
        if program['window'] != self.window or reference['schema_version'] != 'regional-shape-mixture-1.0':
            raise ValueError('Shape learning/control window or schema mismatch')
        self.tau, self.delta = float(program['mixture_temperature']), float(program['robust_delta'])
        self.scale = np.asarray(reference['feature_scale_A2'])
        if not all(math.isfinite(v) and v > 0 for v in (self.tau, self.delta)) or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError('Finite positive shape response/scales required')

    def observables(self, x, atoms, mask, anchor=None):
        if anchor is None or anchor.shape != x.shape:
            raise ValueError('Matched detached endpoint anchor required')
        anchor = anchor.detach()
        if atoms.ndim == 3:
            atoms = atoms.detach().argmax(-1)
        eligible = mask.bool().clone()
        if self.reference['channel'] == 'NOS':
            eligible &= torch.isin(atoms, atoms.new_tensor([self.reference['atom_vocabulary'][a] for a in ('N', 'O', 'S')]))
        valid = eligible.sum(1) >= 2
        safe = torch.where(eligible.any(1)[:, None], eligible, mask.bool())
        columns, core = [], torch.zeros_like(mask, dtype=torch.bool)
        for region in self.reference['regions'].values():
            points = x.new_tensor(region['points_A'])
            distance = (anchor[:, :, None]-points[None, None]).square().sum(-1).clamp_min(1e-20).sqrt().amin(-1)
            logits = (-.5*(distance/self.reference['spatial_width_A']).square()).masked_fill(~safe, -torch.inf)
            w = logits.softmax(1)
            center = (w[..., None]*x).sum(1)
            centered = x-center[:, None]
            tensor = torch.einsum('bn,bni,bnj->bij', w, centered, centered)
            columns.extend([tensor[:, i, j:j+1]*(math.sqrt(2) if i != j else 1.)
                            for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))])
            core |= eligible & (distance <= self.program['core_radius_A'])
        return torch.cat(columns, 1), valid, core

    def __call__(self, x, atoms, mask, time, anchor=None):
        j = int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time) > 2e-6:
            raise ValueError('No exact learned shape time')
        raw, valid, core = self.observables(x.double(), atoms, mask, anchor.double() if anchor is not None else None)
        z = raw/raw.new_tensor(self.scale)
        modes = self.reference['frames'][j]['modes']
        center = z.new_tensor([m['center_scaled'] for m in modes])
        precision = z.new_tensor(np.linalg.inv([m['covariance_dimensionless'] for m in modes]))
        delta = z[:, None]-center[None]
        q = torch.einsum('bmi,mij,bmj->bm', delta, precision, delta).clamp_min(0)/z.shape[1]
        cost = self.delta**2*(torch.sqrt(1+q/self.delta**2)-1)
        logits = -cost/self.tau-math.log(len(modes))
        value = self.tau*torch.logsumexp(logits, 1)
        gate = torch.sqrt(q.amin(1)/(1+q.amin(1)))
        return torch.where(valid, value, z.sum(1)*0), {'observables': raw, 'available': valid,
            'core_mask': core, 'nearest_standardized_rms': q.amin(1).sqrt(),
            'dose_gate': torch.where(valid, gate, 0.), 'mode_responsibilities': logits.softmax(1)}
