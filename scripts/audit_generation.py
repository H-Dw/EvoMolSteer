"""Audit completed HDF5 contents, pairing and selection lineage without unpickling."""
import argparse
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.storage.arrays import array_digest,byte_equal
from evomolsteer.storage.trajectory import TrajectoryPackage


def audit(campaign,output):
    root=Path(campaign);cfg=read_json(root/'config.json');opt=cfg['experiment']
    if read_json(root/'COMPLETE.json')['status']!='complete':raise ValueError('Campaign is incomplete')
    signatures={};rows=[];arms=opt['arms'].split(',')
    for arm in arms:
        for batch in range(opt['n']//opt['batch']):
            directory=root/arm/f'batch_{batch:03d}';path=directory/'trajectory.h5'
            if (directory/'trajectory.npz').exists():raise ValueError('Unexpected redundant raw trajectory')
            if (directory/'trajectory.stages').exists():raise ValueError('Intermediate stages were not retired')
            with TrajectoryPackage(path) as z:
                fields=z.verify()
                sig={k:array_digest(z.read(k,0)) for k in ['current_coords','current_atomics','current_bonds','current_charges','mask']}
                if batch in signatures and signatures[batch]!=sig:raise ValueError('Initial state pairing mismatch')
                signatures[batch]=sig
                counts=z.read('offspring_count');ids=z.read('selected_indices');resampled=z.read('resampled').astype(bool)
                times=z.read('score_time')[:,0]
                for i in range(len(ids)):
                    if not np.array_equal(counts[i],np.bincount(ids[i],minlength=opt['batch'])):raise ValueError('Offspring counts disagree')
                    if not resampled[i] and not np.array_equal(ids[i],np.arange(opt['batch'])):raise ValueError('Unmarked resampling')
                current,proposal=z.read('current_coords'),z.read('proposal_coords')
                for i in range(len(ids)-1):
                    if not byte_equal(current[i+1],proposal[i][ids[i]]):raise ValueError('Broken coordinate ancestry')
                if arm.startswith('gradient') and resampled.any():raise ValueError('Gradient arm used selection')
                if arm=='single' and not np.array_equal(np.flatnonzero(resampled),np.arange(51)):raise ValueError('Unexpected selection schedule')
                records=read_json(directory/'final_records.json')
                if len(records)!=opt['batch']:raise ValueError('Missing successful/failed final candidates')
                rows.append({'arm':arm,'batch':batch,'trajectory_sha256':digest(path),'verified_fields':fields,
                    'steps':len(ids),'candidate_slots_per_step':len(ids[0]),'resampling_steps':np.flatnonzero(resampled).tolist(),
                    'failed_final_candidates_retained':sum(not r['build_success'] for r in records),
                    'coordinate_parent_links_verified':(len(ids)-1)*opt['batch']})
    result={'status':'passed','all_initial_states_paired':True,'initial_state_signatures':signatures,
            'trajectory_audits':rows,'no_original_npz_created':True,'code_commit':cfg['extension']['code_commit']}
    write_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=audit(a.campaign,a.output);print('Verified',len(r['trajectory_audits']),'complete trajectories and all initial-state pairing')
