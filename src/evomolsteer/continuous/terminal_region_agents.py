"""Real API regional analysis and deterministic compilation of local programs."""
import copy
import gzip
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import jsonschema

from ..io import clean,digest,read_json,write_json
from ..deepseek_api import call_json
from ..agent_workflow.engine import run_agent
from ..agent_workflow.sources import EvidenceTools
from ..agent_workflow.contracts import obj,strict_json
from ..generation.window_reference import load_reference

REGION=re.compile(r'landmark_(\d+)_')


def prepare_evidence(mining, output):
    mining,output=Path(mining),Path(output)
    manifest=read_json(mining/'manifest.json')
    effects=pd.read_parquet(mining/'whole_window_effects.parquet')
    donor=effects[effects.split.eq('discovery')].copy()
    donor['coordinate_feature']=donor.feature
    donor['feature']=donor.measure+'/'+donor.feature
    donor=clean(donor.to_dict('records'))
    regions=[]
    for region in range(manifest['regions']):
        rows=[r for r in donor if r['region_index']==region]
        association=sorted([r for r in rows if r['measure']=='future_partial_correlation'],
            key=lambda r:(r['q'], -abs(r['effect'] or 0),r['feature']))
        contrast=sorted([r for r in rows if r['measure']=='elite_contrast'],
            key=lambda r:(r['q'], -abs(r['effect'] or 0),r['feature']))
        regions.append({'region_index':region,'association_candidates':association[:2],
            'elite_contrast_candidate':contrast[:1]})
    trends=read_json(mining/'empirical_trends.json')
    functions=trends['functions']
    reference=load_reference(mining/'terminal_teacher_reference.json.gz')
    teachers=np.asarray(reference['frames'][-1]['teacher_endpoint_A'])
    points=np.asarray(reference['landmarks_A'])
    for region in regions:
        r=region['region_index']; d2=((teachers-points[r])**2).sum(-1)
        region['teacher_spatial_support']={str(radius):{'supported_teachers':int((d2.min(1)<radius**2).sum()),
            'mean_teacher_atoms_inside':float((d2<radius**2).sum(1).mean())}
            for radius in manifest['config']['agent']['allowed_radii_A']}
    columns=list(donor[0])
    packet={'objective':{'primary':'Final predicted target pIC50','secondary':['strain','structural validity'],
        'teacher_selection':'Top-k valid decoded final molecules in the configured discovery panel',
        'selection_config':manifest['config']['elite_selection']},
        'clocks':{'learning_window':manifest['window'],'observed_score_times':manifest['times'],
            'terminal_label_time':1.,'coordinate_representation':'Predicted endpoint world coordinates at actual ancestor score time'},
        'teacher_modes':{'regions':regions,'terminal_elites':manifest['terminal_elites'],
            'unique_teacher_ancestors':manifest['unique_teacher_ancestors'],'split':manifest['split'],
            'terminal_elite_quality':clean(pd.read_csv(mining/'terminal_elites.csv')[[c for c in ['batch','slot','pic50_on_rescore','mmff_relief_per_heavy','energy_status','pb_fast_pass'] if c in pd.read_csv(mining/'terminal_elites.csv').columns]].to_dict('records'))},
        'prerequisites':{'minimum_curves':manifest['config']['agent'].get('minimum_curves',3),
            'minimum_compared_regions':manifest['config']['agent'].get('minimum_compared_regions',3),
            'validation_evidence':'Withheld from role-based selection; no outcome from tested generation is supplied'},
        'limitations':manifest['limitations'],
        'tables':{'continuous_trends':{'columns':columns,'rows':[[r[k] for k in columns] for r in donor]}}}
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'packet.json',packet)
    write_json(output/'index.json',{'ids':[r['evidence_id'] for r in donor]})
    write_json(output/'continuous_functions.json',{'window':manifest['window'],'times':manifest['times'],'functions':functions})
    write_json(output/'observable_definitions.json',{
        'regional_coordinates':{'fields':manifest['feature_names'],'radius_A':manifest['config']['analysis']['radius_A'],
            'frame':'Fixed receptor world frame; no separate rigid alignment',
            'elite_contrast':'Root-balanced elite-ancestor minus observed ordinary-ancestor feature, standardized by node variance',
            'future_partial_correlation':'Root-balanced rank correlation with observed descendant final scores, controlling instantaneous joint head'},
        'teacher_reward':{'formula':'tau*logsumexp(log_prior-pseudo_Huber(region_weighted_matched_MSE)/tau)',
            'membership':'Compact teacher-based spatial weights; zero outside selected regions',
            'gradient':'Conditional endpoint loss pulled through real FLOWR endpoint VJP',
            'no_per_step_affinity_call':True}})


class RegionTools(EvidenceTools):
    def __init__(self,root):
        super().__init__(root)
        self.schemas['read_region_catalogue']=obj({})
    def _read_region_catalogue(self):
        result=[]
        for i,region in enumerate(self.packet['teacher_modes']['regions']):
            result.append(self._deliver('region_catalogue/'+str(i),'packet.json',f'/teacher_modes/regions/{i}',region,kind='region_catalogue'))
        return {'regions':result}
    def _source_value(self,identity):
        if identity['kind']=='region_catalogue':
            return self.packet['teacher_modes']['regions'][int(identity['source_id'].split('/')[-1])]
        return super()._source_value(identity)
    def resolve(self,refs):
        normal=[v for v in refs if self.delivered[v]['kind']!='region_catalogue']
        result=super().resolve(normal)
        for ref in refs:
            identity=self.delivered[ref]
            if identity['kind']=='region_catalogue':
                result.append(identity['source_id'])
        return list(dict.fromkeys(result))


def finish_analysis(result,tools,limits):
    gaps=[]
    curves={v['source_id'][6:] for v in tools.delivered.values() if v['kind']=='curve'}
    regions={int(REGION.search(v).group(1)) for v in curves if REGION.search(v)}
    if not any(v['kind']=='region_catalogue' for v in tools.delivered.values()):
        gaps.append({'reason':'Read the complete region catalogue using read_region_catalogue.'})
    if len(curves)<limits.get('minimum_curves',3) or len(regions)<limits.get('minimum_compared_regions',3):
        gaps.append({'reason':'Insufficient distinct regional curve comparisons','curves':len(curves),'regions':len(regions)})
    for h in result['hypotheses']:
        feature=h.get('observable','')
        refs=h['evidence_refs']
        identities=[tools.delivered[r] for r in refs]
        if feature not in curves or not REGION.search(feature) or not any(v['kind']=='curve' and v['source_id']=='curve/'+feature for v in identities):
            gaps.append({'reason':'Hypothesis observable must be an exact retrieved curve key and cite that curve','observable':feature})
    return gaps


def run_roles(repo,mining,output,*,caller=call_json):
    repo,mining,output=map(Path,(repo,mining,output))
    manifest=read_json(mining/'manifest.json'); config=manifest['config']; limits=config['agent']
    evidence=output/'evidence';prepare_evidence(mining,evidence)
    tools=RegionTools(evidence)
    instructions=(repo/'skills/analyst/SKILL.md').read_text(encoding='utf-8')+'\n'+(repo/'skills/terminal-region-analyst/SKILL.md').read_text(encoding='utf-8')
    task={'objective':'Identify final-affinity-associated regions for local teacher guidance',
        'scope':'Whole learned coordinate window; final labels through actual ancestry only',
        'coverage':{'minimum_curves':limits.get('minimum_curves',3),'minimum_compared_regions':limits.get('minimum_compared_regions',3)},
        'observable_contract':'Each hypothesis.observable must equal the exact get_curve feature key, including measure/ prefix.',
        'tools_note':'Start with read_region_catalogue; it delivers every region with association and elite contrast statistics. Compare actual curves, including a competing candidate.'}
    analyst=run_agent('Analyst',tools,instructions,task,output/'Analyst',
        max_steps=limits['max_steps'],max_output_tokens=limits['max_output_tokens'],caller=caller,
        finish_validator=lambda result,t:finish_analysis(result,t,limits),finish_policy='terminal-region-evidence-1.0')
    hypotheses={h['id']:h for h in analyst['hypotheses']}
    region_hypotheses={}
    for identifier,h in hypotheses.items():
        match=REGION.search(h.get('observable',''))
        if match:region_hypotheses.setdefault(int(match.group(1)),[]).append(identifier)
    if not region_hypotheses:
        write_json(output/'selection.json',{'status':'deferred','reason':'Analyst has no defensible regional hypothesis','analyst_sha256':digest(output/'Analyst/Analyst.json')})
        return None
    item=obj({'region_index':{'type':'integer','enum':sorted(region_hypotheses)},
        'weight':{'type':'number','exclusiveMinimum':0,'maximum':1},
        'hypothesis_ids':{'type':'array','items':{'enum':sorted(hypotheses)},'minItems':1,'uniqueItems':True},
        'evidence_ids':{'type':'array','items':{'enum':sorted({v for h in hypotheses.values() for v in h['evidence_ids']})},'minItems':1,'uniqueItems':True}})
    schema=obj({'role':{'const':'Designer'},'decision':{'enum':['local','defer']},
        'selected_regions':{'type':'array','items':item,'maxItems':limits['maximum_regions']},
        'radius_A':{'type':'number','enum':limits['allowed_radii_A']},'rationale':{'type':'string'},
        'confidence':{'enum':['exploratory','supported']},'failure_modes':{'type':'array','items':{'type':'string'}}})
    def validate_design(result):
        selected=result['selected_regions']
        if result['decision']=='local' and not selected:raise jsonschema.ValidationError('Local design needs a selected region')
        if result['decision']=='defer' and selected:raise jsonschema.ValidationError('Deferred design cannot activate regions')
        if len(set(r['region_index'] for r in selected))!=len(selected):raise jsonschema.ValidationError('Duplicate selected region')
        for r in selected:
            if not set(r['hypothesis_ids'])<=set(region_hypotheses[r['region_index']]):
                raise jsonschema.ValidationError('Selected region must link to its actual Analyst hypothesis')
            permitted={v for h in r['hypothesis_ids'] for v in hypotheses[h]['evidence_ids']}
            if not set(r['evidence_ids'])<=permitted:raise jsonschema.ValidationError('Selected region evidence is outside its cited hypotheses')
    designer_instructions=(repo/'skills/designer/SKILL.md').read_text(encoding='utf-8')+'\n'+(repo/'skills/terminal-region-designer/SKILL.md').read_text(encoding='utf-8')
    messages=[{'role':'system','content':designer_instructions},
        {'role':'user','content':json.dumps({'handoff':analyst,'region_hypotheses':region_hypotheses,
            'contract':{'registry':'endpoint_teacher_region','background_weight':0,'maximum_regions':limits['maximum_regions'],
                'allowed_radii_A':limits['allowed_radii_A'],'affinity_head_gradient':False,'score_beta':0,
                'teacher_selection':'Configured terminal top-k; static forecast geometry from actual last-node ancestors',
                'dose_and_temperatures':'Frozen M2 defaults','hypothesis_to_region':'Regions must be linked to the matching hypothesis, not just any existing citation'},
            'evidence':clean(tools.packet['teacher_modes']),'definitions':tools.definitions},ensure_ascii=False)}]
    design=caller(messages,schema,output/'Designer.json',input_store=output/'inputs',
        max_output_tokens=limits['max_output_tokens'],response_validator=validate_design,response_parser=strict_json,
        format_policy='terminal-region-designer-json-1.0')
    validate_design(design)
    selection={'status':'selected' if design['decision']=='local' else 'deferred','design':design,
        'analyst_sha256':digest(output/'Analyst/Analyst.json'),'designer_sha256':digest(output/'Designer.json'),
        'instruction_files':{p:digest(repo/p) for p in ['skills/analyst/SKILL.md','skills/designer/SKILL.md',
            'skills/terminal-region-analyst/SKILL.md','skills/terminal-region-designer/SKILL.md']},
        'actual_curves_read':[v['source_id'] for v in tools.delivered.values() if v['kind']=='curve'],
        'candidate_regions':sorted(region_hypotheses),'no_synthetic_fallback':True}
    write_json(output/'selection.json',selection)
    return design if design['decision']=='local' else None


def compile_programs(repo,mining,selection,output,base_program,*,windows=None):
    repo,mining,output=[Path(p).resolve() for p in (repo,mining,output)];manifest=read_json(mining/'manifest.json')
    config=manifest['config']; source=load_reference(mining/'terminal_teacher_reference.json.gz')
    base=read_json(repo/base_program); output.mkdir(parents=True,exist_ok=True)
    design=read_json(selection)['design'] if selection is not None else None
    if design is not None and design['decision']!='local':design=None
    specs=[]
    for window in windows or config['guidance_windows']:
        start,end=map(float,window)
        if not source['window'][0]<=start<end<=source['window'][1]:raise ValueError('Control window must be a subset of learned support')
        reference=copy.deepcopy(source)
        reference['window']=[start,end]
        reference['frames']=[f for f in reference['frames'] if f['time']>=start-2e-6 and f['time']<end-2e-6]
        reference['times']=[f['time'] for f in reference['frames']]
        reference['source_learning_window']=source['window']
        if not reference['frames']:raise ValueError('Empty controlled score grid')
        ref_path=output/f'reference_{start:g}_{end:g}.json.gz'
        ref_path.write_bytes(gzip.compress(json.dumps(reference,separators=(',',':'),allow_nan=False).encode(),mtime=0))
        for mode in ['global']+(['local'] if design is not None else []):
            program=copy.deepcopy(base)
            program.update(program_id=f'terminal_topk_{mode}_{start:g}_{end:g}',window=[start,end],reference_sha256=digest(ref_path),
                teacher_selection='Decoded terminal top-k; actual parent-index ancestry',source_learning_window=source['window'])
            if mode=='local':
                weights=np.zeros(manifest['regions'])
                for region in design['selected_regions']:weights[region['region_index']]=region['weight']
                weights/=weights.sum()
                program.update(reward_view='endpoint_teacher_region',coordinate_region_weights=weights.tolist(),
                    coordinate_region_radius_A=design['radius_A'],coordinate_background_weight=0,
                    agent_selection_sha256=digest(selection))
            program_path=output/f'{program["program_id"]}.json'
            write_json(program_path,program)
            specs.append({'id':program['program_id'],'mode':mode,'window':window,
                'program':program_path.relative_to(repo).as_posix(),'reference':ref_path.relative_to(repo).as_posix(),
                'program_sha256':digest(program_path),'reference_sha256':digest(ref_path)})
    write_json(output/'programs.json',{'programs':specs,'config':config,'selection_sha256':digest(selection) if selection else None,
        'formula':'M2 robust multimodal coordinate loss; local mode has zero outside-region endpoint loss'})
    return specs
