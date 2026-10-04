"""Deterministic regional atom/graph observables, without inventing energies.

Hard intermediate labels are categorical model states, not sanitized molecules.
Coordinates of contact centroids are offsets in the supplied aligned receptor
frame. Slot indices are not assumed to be chemical identities across time.
"""
import numpy as np
from scipy.special import expit


def extend_catalog(catalog, charge_vocabulary, cfg):
    catalog['schema_version'] = '3.0'
    catalog['charge_vocabulary'] = charge_vocabulary
    # The input must explicitly declare the upstream category semantics.
    orders = cfg.get('bond_order_by_label')
    if orders != {'0': 0., '1': 1., '2': 2., '3': 3., '4': 1.5}:
        raise ValueError('Verified FLOWR bond label mapping required')
    catalog['bond_order_by_label'] = orders
    def add(region, kind, unit, interpretation):
        fid = region + '::' + kind
        catalog['features'][fid] = dict(id=fid, region=region, kind=kind, unit=unit,
            elements=[], differentiable_supported=False, interpretation=interpretation)
    for element in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', '<PAD>']:
        add('ligand', 'atom_fraction_' + element, 'fraction', 'Fraction of active slots carrying this hard category')
    for kind, unit in [('charge_valid_fraction','fraction'), ('formal_charge_mean','e'),
                       ('formal_charge_abs_mean','e'), ('bond_density','fraction'),
                       ('bond_order_mean','order'), ('bond_length_mean','A'),
                       ('bond_length_sd','A'), ('bond_asymmetry_fraction','fraction')]:
        add('ligand', kind, unit, 'Raw graph observable; no sanitization or force-field energy interpretation')
    for label in range(1,5):
        add('ligand', f'bond_label_{label}_fraction', 'fraction', 'Among nonzero upper-triangle bond labels')
    for region in catalog['regions']:
        for axis in 'xyz':
            add(region, 'contact_centroid_offset_' + axis, 'A',
                'Smooth contact-weighted ligand centroid minus receptor-region centroid; aligned frame, not rotation invariant')
        for kind, unit, note in [
            ('contact_hetero_fraction','fraction','Contact-weighted fraction of N/O/S categories; not hydrogen bonding'),
            ('contact_charge_mean','e','Contact-weighted formal charge category, not partial charge or electrostatic energy'),
            ('contact_bond_order_mean','order','Contact-weighted mean incident bond order on ligand atoms'),
            ('steric_overlap_proxy','A^2','Mean squared positive part of (2 A - atom distance); geometric proxy, not physical energy')]:
            add(region, kind, unit, note)
    catalog['energy_status'] = {'physical_energy_available': False,
        'reason': 'No force-field atom typing/partial charges/hydrogens or per-region energies in the recorded trajectory.',
        'proxy': 'steric_overlap_proxy has units A^2 and must never be labeled kcal/mol.',
        'affinity_score': 'pic50_on is a model prediction, not a measured or decomposed energy.'}
    return catalog


def measure_chemistry(xyz, atomics, mask, bonds, charges, catalog, chunk_size=256):
    xyz = np.asarray(xyz, float); mask = np.asarray(mask, bool)
    vocab = catalog['atom_vocabulary']; n = mask.sum(1)
    result = {}
    def average(value, weights):
        valid = np.isfinite(value)
        den = np.sum(weights * valid, axis=1)
        return np.divide(np.sum(np.where(valid,value,0)*weights,axis=1),den,
                         out=np.full(len(value),np.nan),where=den>0)
    for element in ['C','N','O','S','F','Cl','Br','I','<PAD>']:
        result['ligand::atom_fraction_'+element] = average(atomics==vocab[element],mask)
    lookup = np.full(max(catalog['charge_vocabulary'].values())+1, np.nan)
    for charge, label in catalog['charge_vocabulary'].items():
        if charge != '<PAD>': lookup[label] = float(charge)
    if charges.max() >= len(lookup): raise ValueError('Unknown charge category')
    q = lookup[charges]
    result['ligand::charge_valid_fraction'] = average(np.isfinite(q),mask)
    result['ligand::formal_charge_mean'] = average(q,mask)
    result['ligand::formal_charge_abs_mean'] = average(np.abs(q),mask)
    pm = mask[:,:,None] & mask[:,None,:]
    upper = pm & np.triu(np.ones(bonds.shape[1:],bool),1)
    valid_pairs = upper.sum((1,2))
    edge = upper & (bonds>0)
    count = edge.sum((1,2))
    order_lookup = np.array([catalog['bond_order_by_label'][str(i)] for i in range(5)])
    if bonds.max()>4: raise ValueError('Unknown bond label')
    order = order_lookup[bonds]
    dist = np.linalg.norm(xyz[:,:,None,:]-xyz[:,None,:,:],axis=-1)
    def edgesum(value):
        return np.divide(np.where(edge,value,0).sum((1,2)),count,
                         out=np.full(len(xyz),np.nan),where=count>0)
    result['ligand::bond_density'] = count/np.maximum(valid_pairs,1)
    result['ligand::bond_order_mean'] = edgesum(order)
    mean = edgesum(dist)
    result['ligand::bond_length_mean'] = mean
    result['ligand::bond_length_sd'] = np.sqrt(np.maximum(edgesum(dist**2)-mean**2,0))
    result['ligand::bond_asymmetry_fraction'] = ((bonds!=bonds.transpose(0,2,1)) & upper).sum((1,2))/np.maximum(valid_pairs,1)
    for label in range(1,5): result[f'ligand::bond_label_{label}_fraction'] = edgesum(bonds==label)
    # Symmetric upper-triangle graph; asymmetry remains a separately reported QC.
    symmetric_order = np.where(edge,order,0); symmetric_order += symmetric_order.transpose(0,2,1).copy()
    degree = (symmetric_order>0).sum(2)
    atom_order = np.divide(symmetric_order.sum(2),degree,out=np.full(mask.shape,np.nan),where=degree>0)
    for region, spec in catalog['regions'].items():
        points = np.asarray(spec['points_A']); center = points.mean(0)
        collected = {k:[] for k in ['contact_centroid_offset_'+a for a in 'xyz'] +
            ['contact_hetero_fraction','contact_charge_mean','contact_bond_order_mean','steric_overlap_proxy']}
        for start in range(0,len(xyz),chunk_size):
            sl = slice(start,start+chunk_size); xx=xyz[sl]; mm=mask[sl]
            distances = np.linalg.norm(xx[:,:,None,:]-points[None,None,:,:],axis=-1)
            w = expit((catalog['contact_midpoint_A']-distances.min(2))/catalog['contact_width_A'])*mm
            for k,axis in enumerate('xyz'):
                collected['contact_centroid_offset_'+axis].append(average(xx[:,:,k]-center[k],w))
            collected['contact_hetero_fraction'].append(average(np.isin(atomics[sl],[vocab[e] for e in ['N','O','S']]),w))
            collected['contact_charge_mean'].append(average(q[sl],w))
            collected['contact_bond_order_mean'].append(average(atom_order[sl],w))
            overlap = np.maximum(2.-distances,0)**2
            collected['steric_overlap_proxy'].append(average(overlap.mean(2),mm))
        for kind, values in collected.items(): result[region+'::'+kind] = np.concatenate(values)
    return result
