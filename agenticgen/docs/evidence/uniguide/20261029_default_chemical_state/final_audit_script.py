"""Read-only verification of the completed chemical-state engineering acceptance."""
from pathlib import Path
import json, hashlib, datetime, subprocess, fcntl
import h5py
import numpy as np
import torch
from agenticgen.uniguide.native_tools import load_pt, tensor_digest
from agenticgen.uniguide.geometry_chemistry import match_fractions

root=Path('/workspace/agenticgen_chemical_default_acceptance_20261029_b49adddc_r2')
code=Path('/workspace/uniguide_audits/chemical-default-b49adddc')
audit=Path('/workspace/uniguide_audits/chemical-default-b49adddc-r2')
def read(p):return json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
assert (root/'FINISHED.json').exists()
assert not (audit/'FAILED.json').exists()
complete=read(root/'acceptance/COMPLETE.json')
protected=read(root/'SOURCE_PRESERVATION.json')['source_hashes']
changed=[p for p,h in protected.items() if digest(Path(p))!=h]
assert not changed,changed
checks={}
for teacher,accepted in complete['checks'].items():
    directory=root/'acceptance'/teacher
    batches=[]
    for name in ['separate_0','separate_1','mixed_twenty','mixed_resumed']:
        folder=directory/'candidate_batch'/name
        terminal=load_pt(folder/'terminal.pt.gz')
        state=load_pt(folder/'checkpoint_100.pt.gz')
        raw=state['raw_output_prediction']['target']
        chemical=read(folder/'CHEMICAL_CONDITION.json')
        output=read(folder/'OUTPUT_POLICY.json')
        assert not output['coordinate_projection'] and not output['chemical_projection']
        assert not output['extra_terminal_forward']
        assert output['chemical_state_projection']
        assert output['chemical_condition_sha256']==chemical['chemical_condition_sha256']
        binding_key='candidate_binding' if name.startswith('mixed_') else 'geometry_binding'
        assert state[binding_key]['chemical_binding']==chemical
        assert state['raw_output_prediction']['last_loop']==output['last_loop']
        assert output['last_loop']['step']==99
        raw_channels={k:torch.equal(raw[k],terminal[k]) for k in chemical['features']}
        assert all(raw_channels.values()),raw_channels
        ids=torch.tensor(chemical['generated_slots'])[None].expand(len(terminal['mask']),-1)
        features={k:torch.tensor(v) for k,v in chemical['features'].items()}
        final_matches=match_fractions(terminal,ids,features)
        with h5py.File(folder/'trace.h5') as trace:
            coordinates_exact=np.array_equal(terminal['coords'].numpy(),trace['endpoint_A'][99])
            assert coordinates_exact
            post_chemistry_coordinates_exact=np.array_equal(state['current']['coords'].numpy(),trace['guided_proposal'][99])
            assert post_chemistry_coordinates_exact
            delta=np.linalg.norm(trace['guidance_delta_A'][:].astype(np.float64),axis=-1)
            maximum=float(delta.max())
            assert maximum<=output['maximum_single_atom_guidance_increment_angstrom']
            assert not delta[:10].any()
        batches.append(dict(group=name,candidates=len(terminal['mask']),
            raw_chemical_channels_exact=raw_channels,raw_coordinates_exact=bool(coordinates_exact),
            post_chemistry_coordinates_unchanged=bool(post_chemistry_coordinates_exact),
            final_raw_controlled_chemical_match_fraction=final_matches,
            maximum_single_atom_guidance_increment_angstrom=maximum,
            last_loop=output['last_loop'],chemical_condition_sha256=chemical['chemical_condition_sha256']))
    checks[teacher]=dict(batches=batches,acceptance=accepted)
locks={}
for path in [Path('/workspace/agenticgen_uniguide_flowr_20261026/.executor.lock'),root/'.executor.lock']:
    with path.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[str(path)]='free'
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()=='b49adddce93706aca57698e6e9878551b57dab6f'
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
assert not (root/'attempts.json').exists()
assert not (root/'calibration').exists()
files={str(p.relative_to(root)):digest(p) for pattern in ['acceptance/COMPLETE.json','acceptance/*/candidate_batch/COMPLETE.json','GEOMETRY_CONFIG.json','SUBJECTS.json','conditions/*.json'] for p in root.glob(pattern)}
result=dict(time_asia_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    commit='b49adddce93706aca57698e6e9878551b57dab6f',root=str(root),checks=checks,
    protected_files=len(protected),protected_files_changed=changed,locks=locks,code_clean=True,
    source_receipt_hashes=files,formal_or_calibration_started=False,llm_calls=0,
    engineering_only=True,fixture_strengths_not_fitted_multipliers=True,
    earlier_prepare_rejection_preserved='/workspace/agenticgen_chemical_default_acceptance_20261029_b49adddc')
(root/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(time_asia_shanghai=result['time_asia_shanghai'],checks=checks,protected_files_unchanged=len(protected),locks=locks)))
