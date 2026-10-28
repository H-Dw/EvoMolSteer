"""Deterministic synthetic API fixtures, never measured molecular experiments."""
from pathlib import Path
from agenticgen.uniguide.io import frozen,sha,value_sha
from agenticgen.uniguide.geometry_condition import DISTANCE,CONTROL_DISTANCE,NEIGHBOR_DISTANCE,MAX_ERROR
from agenticgen.uniguide.geometry_protocol import NORMALIZED,EFFECTIVE,MULTIPLIER,POLICY,WINDOWS


def build(root):
    root=Path(root);teacher="synthetic_interface_fixture"
    compatibility=dict(teacher_id=teacher,condition_sha256="synthetic-condition",
        random_streams="synthetic-fixed-panel-not-generated",output_policy="fixture")
    history=[]
    for number,(schedule,base) in enumerate([([0,0,0,0],.35),([1,0,0,0],.25),([0,-1,0,0],.4)]):
        run_id=f"fixture/{number}";folder=root/run_id
        values=[base+i*.0001 for i in range(50)]
        if number==1:values[-1]=12.
        rows=[dict(node_id=f"fixture_{i}",logical_seed=i//10,logical_slot=i%10,tensor_row=i,
            build_success=True,connected=True,**{DISTANCE:v,CONTROL_DISTANCE:v,NEIGHBOR_DISTANCE:.2,MAX_ERROR:v*2}) for i,v in enumerate(values)]
        row=dict(id=run_id,teacher_id=teacher,condition_sha256="synthetic-condition",complete=True,attempts=50,built=50,connected=50,unique_smiles=0,
            compatibility_sha256=value_sha(compatibility),fingerprint=value_sha(dict(schedule=schedule,population=50)),
            **{NORMALIZED:schedule,EFFECTIVE:[2.*2**x for x in schedule],MULTIPLIER:2.,
               DISTANCE:sum(values)/50,CONTROL_DISTANCE:sum(values)/50,NEIGHBOR_DISTANCE:.2,MAX_ERROR:max(values)*2},
            window_diagnostics=[dict(steps=list(w),increment_vector_rms_A=.1,max_atom_increment_A=12. if number==1 and i==3 else .2,
                gradient_norm_mean=.2,native_increment_ratio_mean=.1) for i,w in enumerate(WINDOWS)])
        frozen(folder/"records.json",rows);frozen(folder/"aggregate.json",row)
        frozen(folder/"manifest.json",dict(identity=dict(compatibility=compatibility)))
        frozen(folder/"steps.json",[dict(step=step,max_atom_increment_A=[.1]*49+[12. if number==1 and step==80 else .1]) for step in range(100)])
        frozen(folder/"COMPLETE.json",dict(files={n:sha(folder/n) for n in ["records.json","aggregate.json","manifest.json","steps.json"]},synthetic=True))
        history.append(row)
    domain=dict(policy=POLICY,teacher_id=teacher,condition_sha256="synthetic-condition",
        sampling_protocol_sha256=value_sha(compatibility),calibration_key=value_sha([teacher,"synthetic-condition",value_sha(compatibility)]),
        calibrated_guidance_multiplier=2.,normalized_minimum=-3,normalized_maximum=3,initial_source="fixture/0")
    domain["domain_sha256"]=value_sha(domain)
    return history,domain
