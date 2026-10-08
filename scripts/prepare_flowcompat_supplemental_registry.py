"""Freeze executable formula registry and scientific context for new role calls."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.continuous.flowcompat_supplemental import _source_closure

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['program','reference','branch-reference','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--repo',default='.');p.add_argument('--context-file',action='append',default=[])
    a=p.parse_args();root=Path(a.repo).resolve();out=Path(a.output)
    if out.exists():raise FileExistsError('Registry is immutable; use a fresh path')
    program=Path(a.program).resolve();reference=Path(a.reference).resolve();branch=Path(a.branch_reference).resolve()
    codes=_source_closure(root,['scripts/generate_flowcompat_v2_flowr.py'])
    codes={Path(path).relative_to(root).as_posix():sha for path,sha in codes.items()}
    def entry(ref):return {'program_path':str(program),'program_sha256':digest(program),
        'reference_path':str(ref),'reference_sha256':digest(ref)}
    write_json(out,{'schema_version':'flowcompat-supplemental-formula-registry-2.0',
        'programs':{'R26':entry(reference),'R26_branch':entry(branch)},
        'formulas':{
            'branch_mixture':{'reference_kind':'branch_mutation','reward_view':'endpoint_branch_mixture',
                'allowed_updates':['branch_mixture.virtual_mass','branch_mixture.direction_sign','branch_mixture.region_weight_mix'],
                'code_files':codes,
                'formula':'tau*logsumexp(log(pi_k)+log(component_mass)-rho_delta(mean_i(w_ki*||Y_i-T_kb_sigma(i)||^2))/tau). Original center mass>=1-alpha; virtual center T_k+sign*min(observed_branch_RMS,.2A)*d_k. Qualification and matching fixed. c is diagnostic support, not displacement probability.',
                'scope':'Declared reference window; two initial nodes without lag-two evidence are baseline. Regional cost support conditions c and valid local mutation.'},
            'native_control_v2':{'reference_kind':'incumbent','reward_view':'endpoint_pointcloud',
                'allowed_updates':['flow_control.parallel_component_scale','flow_control.time_envelope_power','flow_control.jacobian_gain_saturation','native_rms_ratio'],
                'code_files':codes,
                'formula':'Unchanged R26 scalar; either PSD flow-aligned preconditioner, bounded real VJP gain dose factor G/(G+s), or uniform relative dose. Separate scalar raw-gradient dot actual injection from the preconditioner. These are experimental controls, not an exact guided density.'}},
        'context_files':{str(Path(f).resolve()):digest(f) for f in a.context_file},
        'context':{'window':read_json(program)['window'],'primary_objective':'Predicted binding affinity; strain secondary.',
            'novelty':'Conditioned natural mutations and joint teacher modes. Native SDE/endpoint VJP unchanged. No atom-type or graph equality objective.',
            'implementation_before_efficacy':True,'budget_control':'Positive/reversed/empty directions and gain/uniform dose controls are separate trials. Actual cumulative dose can differ despite equal requested ratios.',
            'failure_response':'Retain validated R26 default until independent frozen confirmation succeeds.'}})

if __name__=='__main__':main()
