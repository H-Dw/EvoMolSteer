"""Paired negative-evidence box exclusion, within one 500-attempt teacher ledger."""
import itertools
import math
from pathlib import Path
import numpy as np
import jsonschema
from .io import read, write, frozen, sha, value_sha
from .box_gates import POLICY, mask
from . import linear_scale as scale
from .absolute_coordinate import RMSD
from .coordinate_statistics import summarize, ranking_eligible
from .geometry_protocol import primary_score

BACKGROUNDS = {
    "6pyy_H7S_A_402": {"UniformRef": [50]*6, "BestCombo": [50,70,70,50,50,70]},
    "3w54_RNB_A_504": {"UniformRef": [40]*6, "BestCombo": [30,40,40,40,60,80]},
}
EXPANDED_BACKGROUNDS = {
    '6pyy_H7S_A_402': {'ReferenceCombo':[50,70,70,50,50,70], 'IncumbentCombo':[50,100,40,40,30,50]},
    '3w54_RNB_A_504': {'ReferenceCombo':[30,40,40,40,60,80], 'IncumbentCombo':[30,40,110,110,200,200]},
}


def search_backgrounds(target, count=6):
    """Split only the historical parameter design; never import measurements."""
    if count not in (6,7):raise ValueError('Expanded backgrounds support the six/seven-window releases')
    return {name:list(actual)+(actual[-1:] if count==7 else [])
            for name,actual in EXPANDED_BACKGROUNDS[target].items()}


def profiles(target, backgrounds=None):
    backgrounds=backgrounds or BACKGROUNDS[target]
    count=len(next(iter(backgrounds.values())))
    if any(len(actual)!=count for actual in backgrounds.values()):raise ValueError('Background dimensions differ')
    result = {}
    def add(name, actual, gates=None):
        result[name] = dict(name=name, actual=list(actual), guidance_box_mask=gates or [True]*count)
    add("NativeControl", [0]*count, [False]*count)
    add("ChemistryOnly", [0]*count)
    for label, actual in backgrounds.items():
        add(label, actual)
        for i in range(count):
            off=list(actual);off[i]=0;gates=[True]*count;gates[i]=False
            add(f"{label}_NativeOff_{i}", off, gates)
            lower=list(actual);lower[i]=max(10,actual[i]-20)
            add(f"{label}_LowerRescue_{i}", lower)
    first=next(iter(backgrounds))
    extra=list(backgrounds[first]);extra[0]+=10
    add(first+"_EarlyUpperRescue", extra)
    assert len(result)==4*count+5
    return result


def paired_effect(on, off, *, resamples=1000):
    """Raw paired seed-cluster means. No missing-value deletion or trimming."""
    identity=lambda r:(r["logical_seed"],r["logical_slot"])
    left,right={identity(r):r for r in on},{identity(r):r for r in off}
    if len(left)!=len(on) or len(right)!=len(off) or left.keys()!=right.keys() or not left:
        return dict(status="unavailable",reason="paired slot identity mismatch")
    if any(not isinstance(r.get(RMSD),(int,float)) or not math.isfinite(r[RMSD]) for r in [*on,*off]):
        return dict(status="unavailable",reason="missing/nonfinite measurement")
    keys=sorted(left)
    changes=np.array([left[k][RMSD]-right[k][RMSD] for k in keys])
    seeds=sorted({k[0] for k in keys})
    seed_changes=[float(changes[[k[0]==s for k in keys]].mean()) for s in seeds]
    rng=np.random.default_rng(721904)
    replicates=np.mean(rng.choice(seed_changes,size=(resamples,len(seeds)),replace=True),axis=1)
    native_mean=float(np.mean([r[RMSD] for r in off]))
    threshold=max(.01,.02*native_mean)
    kept,removed=summarize([r[RMSD] for r in on]),summarize([r[RMSD] for r in off])
    filtered=(kept['upper_tail_filtered_mean']-removed['upper_tail_filtered_mean']
              if kept['ranking_eligible'] and removed['ranking_eligible'] else None)
    return dict(status="complete",paired_slots=len(keys),seed_mean_changes=seed_changes,
        positive_seeds=sum(v>0 for v in seed_changes),raw_retention_cost=float(changes.mean()),
        filtered_retention_cost=filtered,material_threshold=threshold,
        descriptive_cluster_interval=np.quantile(replicates,[.025,.975]).tolist(),
        sign="retained minus native-off; positive favors exclusion",
        independent_validation=False,inference="adaptive fixed-seed panel; not a significance claim")


def coarse_support(effect):
    return (effect.get("status")=="complete" and effect["raw_retention_cost"]>=effect["material_threshold"]
            and effect["positive_seeds"]>=3)


def full_support(effect):
    return (coarse_support(effect) and effect["paired_slots"]==50 and effect["positive_seeds"]>=4
            and effect["filtered_retention_cost"] is not None
            and effect["filtered_retention_cost"]>=effect["material_threshold"]
            and effect["descriptive_cluster_interval"][0]>0)


class ExclusionEvidence:
    """Only frozen, source-verified calculations can be submitted as evidence."""
    def __init__(self, folder, data, sources):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.binding=dict(policy=data.get('box_exclusion_policy',POLICY),data_sha256=value_sha(data),sources=sources)
        self.identifier="ev_"+value_sha(self.binding)[:20]
        self.receipt_ids={self.identifier}
        self.data=dict(evidence_id=self.identifier,**data)
        frozen(self.folder/(self.identifier+'.json'),self.data)
        self.definitions=[dict(type="function",function=dict(name="read_exclusion_evidence",
            description="Read verified box contrasts, counterexamples and confirmed masks; never sample.",
            parameters=dict(type="object",properties={},required=[],additionalProperties=False)))]

    def execute(self, name, args):
        if name!="read_exclusion_evidence" or args:raise ValueError("Unknown read-only evidence request")
        for path,digest in self.binding['sources'].items():
            if sha(path)!=digest:raise ValueError("Exclusion evidence source changed")
        return self.data


def select(config, folder, data, sources, budget, phase, provider=None):
    from agenticgen.providers.deepseek_tools import call_with_tools
    from .tool_runtime import assets
    registry=ExclusionEvidence(Path(folder)/'calculations',data,sources)
    eligible=data['eligible_box_ids']
    count=len(config['windows'])
    maximum=data.get('maximum_confirmed_exclusions',2)
    properties=dict(excluded_box_ids=dict(type="array",uniqueItems=True,maxItems=maximum,
        items=dict(type="integer",minimum=0,maximum=count-1)),
        evidence_ids=dict(type="array",minItems=1,maxItems=4,uniqueItems=True,
                          items=dict(type="string",enum=sorted(registry.receipt_ids))),
        reason=dict(type="string",minLength=1,maxLength=3000))
    if phase=='final':
        properties['confirmed_profile_id']=dict(type="string",enum=list(data['confirmed_options']))
    schema=dict(type="object",properties=properties,required=list(properties),additionalProperties=False)
    def validate(result):
        registry.execute('read_exclusion_evidence',{})
        jsonschema.validate(result,schema)
        if set(result['excluded_box_ids'])-set(eligible):raise ValueError("Box lacks complete negative evidence")
        if phase=='final' and sorted(result['excluded_box_ids'])!=data['confirmed_options'][result['confirmed_profile_id']]:
            raise ValueError("Final exclusion must match an actually confirmed 50-candidate mask")
    instruction=assets(config)['box_selection.md'].read_text(encoding='utf-8')
    import json
    messages=[dict(role='system',content=instruction),dict(role='user',content=json.dumps(
        dict(phase=phase,evidence=registry.data,maximum_excluded=maximum,
             no_sampling_tools=True,empty_exclusion_is_valid=True),ensure_ascii=False))]
    frozen(Path(folder)/'prompt_source.json',dict(messages=messages,response_schema=schema))
    return (provider or call_with_tools)(messages,schema,Path(folder)/'decision.json',registry=registry,
        budget=budget,response_validator=validate,**config['llm'])


class Search:
    def __init__(self,config,subject,model,executor,provider=None):
        self.config,self.subject,self.model,self.executor=config,subject,model,executor
        self.target=subject['teacher']['target'];self.root=Path(config['root'])
        self.folder=self.root/'box_exclusion'/self.target
        from .box_search import enabled
        self.expanded=enabled(config)
        self.count=len(config['windows'])
        self.backgrounds=search_backgrounds(self.target,self.count) if self.expanded else BACKGROUNDS[self.target]
        self.labels=list(self.backgrounds)
        self.rows={};self.specs=profiles(self.target,self.backgrounds);self.provider=provider

    def measure(self,name,count):
        from .linear_calibration import evaluate, used
        if (name,count) in self.rows:return self.rows[name,count]
        spec=self.specs[name]
        prior=[r for (n,_),r in self.rows.items() if n==name]
        if any(not r.get('complete') for r in prior):raise ValueError("Failed profile is immutable")
        increment=count-max((r['attempts'] for r in prior),default=0)
        if used(self.root,self.target)+increment>500:raise ValueError("Exclusion pre-experiment cap exhausted")
        local=dict(self.config,guidance_box_mask=spec['guidance_box_mask'])
        write(self.root/'BUDGET_SCOPE.json',dict(stage='calibration',teacher_id=self.target))
        destination=self.folder/'profiles'/name/f'population_{count}'
        row=evaluate(self.model,local,self.subject,dict(schedule=spec['actual'],folder=destination,
                         stage='calibration',count=count),self.executor)
        if isinstance(row,Exception):
            row=dict(id=destination.relative_to(self.root).as_posix(),complete=False,attempts=count,
                     error_type=type(row).__name__,error=str(row),**{scale.EFFECTIVE:spec['actual']})
            frozen(destination/'CALIBRATION_FAILURE.json',row)
        self.rows[name,count]=row
        write(self.folder/'SEARCH_STATE.json',dict(measurements=[dict(name=n,population=c,row=r)
            for (n,c),r in self.rows.items()],consumed_attempts=used(self.root,self.target)))
        return row

    def records(self,name,count):
        row=self.rows.get((name,count)) or self.rows.get((name,50)) or self.rows.get((name,10))
        if not row or not row.get('complete'):return []
        from .geometry_sampling import validate_receipt
        validate_receipt(self.root/row['id'])
        return [r for r in read(self.root/row['id']/'records.json') if r['logical_slot']<count//5]

    def effect(self,on,off,count=10):
        return paired_effect(self.records(on,count),self.records(off,count))

    def sources(self):
        result={}
        for row in self.rows.values():
            if row.get('complete'):
                for name in ['COMPLETE.json','records.json']:
                    p=self.root/row['id']/name;result[str(p)]=sha(p)
        return result

    def run(self):
        from agenticgen.providers.deepseek_tools import RoundBudget
        from .linear_calibration import used
        for name in self.specs:
            row=self.measure(name,10)
            if name in self.labels:
                for count in (20,50):
                    if not row.get('complete') or not ranking_eligible(row):break
                    row=self.measure(name,count)
        backgrounds=[n for n in self.labels if self.rows.get((n,50),{}).get('complete') and ranking_eligible(self.rows[n,50])]
        if not backgrounds:raise ValueError("No eligible complete background; teacher needs evidence")
        primary=min(backgrounds,key=lambda n:primary_score(self.rows[n,50]))
        contrasts={}
        for i in range(self.count):
            effects=[]
            for background in self.labels:
                off=f'{background}_NativeOff_{i}'
                for retained in (background,f'{background}_LowerRescue_{i}'):
                    effects.append(dict(retained=retained,native_off=off,effect=self.effect(retained,off)))
            if i==0:
                first=self.labels[0]
                effects.append(dict(retained=first+'_EarlyUpperRescue',native_off=first+'_NativeOff_0',
                                    effect=self.effect(first+'_EarlyUpperRescue',first+'_NativeOff_0')))
            contrasts[str(i)]=effects
        candidates=[i for i in range(self.count) if len(backgrounds)==2 and all(coarse_support(r['effect']) for r in contrasts[str(i)])]
        candidates.sort(key=lambda i:(-min(r['effect']['raw_retention_cost']/r['effect']['material_threshold']
                                          for r in contrasts[str(i)]),i))
        # Reserve full confirmations and a possible joint gate before claiming exclusions.
        # Seven boxes cost 410 before confirmations, so the 500 cap admits one two-background proof.
        remaining=500-used(self.root,self.target)
        maximum_confirmed=(2 if remaining>=130 else 1 if remaining>=80 else 0)
        candidates=candidates[:maximum_confirmed]
        confirmed={}
        for i in candidates:
            contexts=backgrounds if len(candidates)==1 else [primary]
            proofs=[]
            for background in contexts:
                name=f'{background}_NativeOff_{i}'
                row=self.measure(name,20)
                if row.get('complete') and ranking_eligible(row):self.measure(name,50)
                proofs.append(self.effect(background,name,50))
            confirmed[str(i)]=proofs
        eligible=[i for i in candidates if all(full_support(r) for r in confirmed[str(i)])]
        coverage_probes=[]
        if self.expanded:
            # The widened first-box grid cannot be excluded using only the old coarse doses.
            # Missing low-dose evidence is unknown, never converted into permission to exclude.
            coverage_probes=self.coverage_tail(primary,50 if len(eligible)>1 else 0)
            early_low_support=all(coarse_support(self.effect(f'{label}_EarlyDose_{dose}',f'{label}_NativeOff_0'))
                for label in self.labels for dose in (0,1,2,5))
            if 0 in eligible and not early_low_support:
                eligible.remove(0)
                # A no-longer-admissible joint mask must not strand a reserved 50.
                if len(eligible)<2:
                    coverage_probes+=self.coverage_tail(primary,0)
        evidence=dict(teacher_id=self.target,backgrounds=self.backgrounds,
            background_full50={n:self.rows[n,50] for n in backgrounds},primary_background=primary,
            coarse_contrasts=contrasts,full_contrasts=confirmed,eligible_box_ids=eligible,
            default='not excluded; no benefit proof required',max_attempts=500,
            statistics_scope='paired fixed initial states, no independent validation')
        if self.count==7:
            evidence.update(guidance_windows=self.config['windows'],maximum_confirmed_exclusions=maximum_confirmed,
                confirmation_budget_policy='reserve complete paired proofs before selection; missing evidence retains boxes')
        if self.expanded:
            evidence.update(box_exclusion_policy=self.config['box_exclusion']['policy'],coverage_probes=coverage_probes,
                early_low_dose_support=early_low_support,
                early_exclusion_status='eligible' if 0 in eligible else 'retained: negative evidence incomplete or contradicted')
        frozen(self.folder/'NEGATIVE_EVIDENCE.json',evidence)
        budget=RoundBudget(self.folder/'model_budget',30,491520)
        proposed=select(self.config,self.folder/'proposal',evidence,self.sources(),budget,'proposal',self.provider)
        excluded=sorted(proposed['excluded_box_ids'])
        if len(excluded)>1:
            gates=[i not in excluded for i in range(self.count)]
            actual=[v if gates[i] else 0 for i,v in enumerate(self.backgrounds[primary])]
            self.specs['CombinedExclusion']=dict(name='CombinedExclusion',actual=actual,guidance_box_mask=gates)
            self.measure('CombinedExclusion',50)
        options={}
        for (name,count),row in self.rows.items():
            if count!=50 or not row.get('complete') or not ranking_eligible(row):continue
            removed=[i for i,on in enumerate(self.specs[name]['guidance_box_mask']) if not on]
            if not removed:
                options[name]=[]
            elif len(removed)==1 and removed[0] in eligible:
                options[name]=removed
            elif name=='CombinedExclusion' and full_support(self.effect(primary,name,50)):
                options[name]=removed
        final_evidence=dict(evidence,proposed_exclusions=excluded,confirmed_options=options,
            confirmed_profiles={n:self.rows[n,50] for n in options})
        final=select(self.config,self.folder/'final',final_evidence,self.sources(),budget,'final',self.provider)
        gates=[i not in final['excluded_box_ids'] for i in range(self.count)]
        compatible=[r for (n,c),r in self.rows.items() if c==50 and r.get('complete') and ranking_eligible(r)
                    and self.specs[n]['guidance_box_mask']==gates]
        winner=min(compatible,key=primary_score)
        if self.provider is None:
            receipt=read(self.folder/'final/decision.receipt.json')
            if receipt.get('final_response_mode')!='strict_tool_v1' or not receipt.get('final_tool_call_id'):
                raise ValueError('Selection has no accepted native strict submission receipt')
        selection=dict(teacher_id=self.target,excluded_box_ids=sorted(final['excluded_box_ids']),
            guidance_box_mask=gates,initial_source=winner['id'],initial_actual_schedule=winner[scale.EFFECTIVE],
            initial_performance=primary_score(winner),selection_receipt_sha256=sha(self.folder/'final/decision.receipt.json')
            if (self.folder/'final/decision.receipt.json').exists() else None,
            evidence_sha256=sha(self.folder/'NEGATIVE_EVIDENCE.json'),consumed_attempts=used(self.root,self.target))
        frozen(self.folder/'SELECTION_FROZEN.json',selection)
        return selection,compatible

    def coverage_tail(self,primary,reserve):
        """Spend only the remaining 500 cap; preserve already charged negative confirmations."""
        from .linear_calibration import used
        from .box_search import grids
        available=lambda:500-used(self.root,self.target)-reserve
        extra=[]
        for dose in (0,1,2,5):
            for label in self.labels:
                actual=list(self.backgrounds[label]);actual[0]=dose
                if dose in grids(self.config,self.target)[0]:extra.append((f'{label}_EarlyDose_{dose}',actual))
        actual=list(self.backgrounds[primary])
        for i in range(4,self.count):
            choices=[v for v in grids(self.config,self.target)[i] if v>max(200,actual[i])]
            if choices:actual[i]=next((v for v in choices if v>=250),choices[-1])
        if actual!=self.backgrounds[primary]:extra.append(('JointFrontier',actual))
        measured=[]
        for name,actual in extra:
            if available()<10:break
            if any(s['actual']==actual and all(s['guidance_box_mask']) for s in self.specs.values()):continue
            self.specs[name]=dict(name=name,actual=actual,guidance_box_mask=[True]*self.count)
            row=self.measure(name,10);measured.append(dict(name=name,row=row))
        full=[r for r in measured if r['row'].get('complete') and ranking_eligible(r['row'])]
        if full and available()>=40:
            winner=min(full,key=lambda r:primary_score(r['row']))
            row=self.measure(winner['name'],50);measured.append(dict(name=winner['name'],row=row))
        return measured


def run(config,subjects,model_loader,executor):
    import torch
    from .geometry_sampling import compatibility
    from .calibration_search import displacement
    from .linear_calibration import ensure_limits
    root=Path(config['root']);ensure_limits(config,len(subjects));domains={}
    for subject in subjects:
        target=subject['teacher']['target'];folder=root/'box_exclusion'/target
        if (folder/'SEARCH_CLAIM.json').exists():
            raise RuntimeError("Exclusion attempt already claimed; inspect rather than restart")
        frozen(folder/'SEARCH_CLAIM.json',dict(policy=config['box_exclusion']['policy'],config_sha256=value_sha(config)))
        model=model_loader(config,subject['teacher'])
        try:
            selection,rows=Search(config,subject,model,executor).run()
        finally:
            del model;torch.cuda.empty_cache()
        local=dict(config,guidance_box_mask=selection['guidance_box_mask'])
        binding=compatibility(local,subject['teacher'],subject['condition']);protocol=value_sha(binding)
        key=value_sha([target,subject['condition']['condition_sha256'],protocol])
        winner=next(r for r in rows if r['id']==selection['initial_source'])
        body=dict(scale_policy=scale.OPEN_POLICY,teacher_id=target,condition_sha256=subject['condition']['condition_sha256'],
            sampling_protocol_sha256=protocol,calibration_key=key,control_settings=config['guidance_control'],
            calibrated_guidance_reference_scale=0,guidance_scale_grid_unit=config['guidance_control'][scale.UNIT],
            initial_source=winner['id'],initial_guidance_scale_tick_schedule=[int(x//config['guidance_control'][scale.UNIT]) for x in winner[scale.EFFECTIVE]],
            guidance_windows=config['windows'],bootstrap_sources=[r['id'] for r in rows],
            confirmed_effective_levels=sorted({v for r in rows for v in r[scale.EFFECTIVE]}),
            confirmed_profiles=[dict(source_id=r['id'],actual_schedule=r[scale.EFFECTIVE]) for r in rows],
            constant_comparison=[dict(source_id=r['id'],eligible=True,absolute_displacement=displacement(r)) for r in rows],
            risk_boundary=None,evidence_revision=0,boundary_status='profile_scoped_evidence',
            stability_guarantee=False,global_optimum_claimed=False,source_rows_sha256=value_sha(rows),
            exploration_policy=config['guidance_exploration'],initial_calibration_performance=primary_score(winner),
            box_exclusion_policy=config['box_exclusion']['policy'],guidance_box_mask=selection['guidance_box_mask'],
            selection_sha256=value_sha(selection),admission_policy='declared_grid_and_actual_budget; fixed_native_gates')
        from .box_search import enabled as expanded_search, grids, early_shape
        if expanded_search(config):
            body.update(search_grids=grids(config,target),early_guidance_shape=early_shape(config),
                        execute_all_proposed=True,distance_unit='per_window_grid_index')
        domain=scale.seal(body);scale.validate_domain(domain)
        dest=root/'calibration/frozen'/key
        frozen(dest/'domain.json',domain)
        frozen(dest/'COMPLETE.json',dict(domain_sha256=domain['domain_sha256'],measurement_receipts={
            r['id']:sha(root/r['id']/'COMPLETE.json') for r in rows},selection_sha256=value_sha(selection)))
        domains[target]=domain
    frozen(root/'calibration/domains.json',domains)
    frozen(root/'calibration/TEACHER_STATUS.json',{t:dict(status='ready',domain_sha256=d['domain_sha256']) for t,d in domains.items()})
    frozen(root/'calibration/COMPLETE.json',dict(domain_sha256=value_sha(domains),box_exclusion=True))
    return domains
