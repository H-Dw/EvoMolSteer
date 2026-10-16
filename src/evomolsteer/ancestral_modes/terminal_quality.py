"""Incremental, reusable final-structure assays for verified frozen batches."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json,write_json
from evomolsteer.local_motif.quality import evaluate as posebusters
from evomolsteer.local_motif.evaluation import energy,score_records,paired


def analyze(repo,config,protein,output):
    repo=Path(repo);m=read_json(config);root=repo/m['generated_root'];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    state=read_json(repo/m['campaign_study']/'state.json');tau=state['original']['tau'];rows=[]
    for batch in sorted(root.glob('R3[45]_v3_b*')):
        if not (batch/'download_verified.json').exists():continue
        guide=batch/batch.name;unguided=batch/'unguided'
        if not (guide/'COMPLETE.json').exists() or not (unguided/'COMPLETE.json').exists():continue
        pairing=paired(guide,unguided,tau)
        for arm,folder in [('guidance',guide),('unguided',unguided)]:
            tag=batch.name+'__'+arm
            pb=posebusters(folder,protein,output/(tag+'.posebusters.json'));e=energy(folder,output/(tag+'.energy.json'))
            rows.append({'batch_label':batch.name,'arm':arm,'result_folder':folder.relative_to(repo).as_posix(),
                'metrics':score_records(read_json(folder/'records.json'),tau),
                'posebusters':{k:pb[k] for k in ('n_attempts','n_evaluated','n_pass_all','n_unresolved','version','protein_sha256')},
                'energy':{k:v for k,v in e.items() if k!='rows'},'paired':pairing})
            print({'quality_batch':batch.name,'arm':arm,'n':e['n'],'PB_pass':pb['n_pass_all'],'energy_median':e['median']},flush=True)
    result={'rows':rows,'completed_paired_batches':len(rows)//2,'scope':'Posthoc same-assay diagnostics; no generation, affinity inference or reward tuning.'}
    write_json(output/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);p.add_argument('--protein',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=analyze(Path(a.repo).resolve(),a.config,Path(a.protein),a.output);print({'completed_paired_batches':r['completed_paired_batches']})


if __name__=='__main__':main()
