"""Install a Git-fetched commit into an isolated runtime, preserving audited dependencies."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

BASE = Path('/opt/MolSteer')
RESULT_ROOT = BASE/'generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022'
NATIVE_PYTHON = '/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--from-runtime', required=True)
    args = parser.parse_args(); revision = args.commit
    old = Path(args.from_runtime).resolve()
    if not re.fullmatch('[0-9a-f]{40}', revision) or old.parent != BASE/'runtimes':
        raise ValueError('Use a full commit and existing audited runtime')
    new = BASE/'runtimes'/('agenticgen_'+revision[:8])
    dep = BASE/'operations'/('agenticgen_deploy_'+revision[:8]); dep.mkdir(parents=True, exist_ok=True)
    def run(command, log, env=None):
        with (dep/log).open('wb') as output:
            status = subprocess.run(command, cwd='/tmp', env=env, stdout=output, stderr=subprocess.STDOUT)
        if status.returncode: raise RuntimeError('Installation failed; inspect '+str(dep/log))
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def git(*a): return subprocess.check_output(['git','-C',str(old),*a],text=True).strip()
    with (RESULT_ROOT/'supervisor.lock').open('a') as supervisor, (RESULT_ROOT/'worker.lock').open('a') as worker:
        for handle in (supervisor,worker): fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('workers.json','launch.json'):
            p=RESULT_ROOT/name
            if p.exists() and (Path('/proc')/str(json.loads(p.read_text())['pid'])).exists():
                raise ValueError('Previous worker exists; no deployment during active work')
        if git('rev-parse',revision+'^{commit}') != revision:
            raise ValueError('Fetch the exact commit from GitHub before installation')
        protected = [RESULT_ROOT/n for n in ('PREPARED.json','config.json','ACCEPTANCE_COMPLETE.json','ACTIVE_PROTOCOL.json')]
        protected += list((RESULT_ROOT/'protocols').glob('*/*.json'))
        protected += list((RESULT_ROOT/'shared_r0').glob('*/COMPLETE.json'))
        protected += list(RESULT_ROOT.glob('FINALIZED_v*.json'))
        for final in RESULT_ROOT.glob('FINALIZED_v*.json'):
            version=final.stem.removeprefix('FINALIZED_')
            protected += [RESULT_ROOT/'versions'/version/n for n in ('binding.json','report.json','review/manifest.json')]
        before={str(p):sha(p) for p in protected}
        if not new.exists():
            run(['git','-C',str(old),'worktree','add','--detach','--no-checkout',str(new),revision],'worktree.log')
            run(['git','-C',str(new),'sparse-checkout','set','--no-cone','/agenticgen/',
                 '/src/evomolsteer/generation/','/src/evomolsteer/adaptive_schedule/vendor/'],'sparse.log')
            run(['git','-C',str(new),'read-tree','-mu','HEAD'],'checkout.log')
        if subprocess.check_output(['git','-C',str(new),'rev-parse','HEAD'],text=True).strip()!=revision:
            raise ValueError('Runtime path already belongs to another revision')
        if subprocess.check_output(['git','-C',str(new),'status','--porcelain','--untracked-files=no']).strip():
            raise ValueError('Runtime checkout is modified')
        venv=new/'agenticgen/.venv'
        if not (venv/'bin/python').exists(): run([NATIVE_PYTHON,'-m','venv',str(venv)],'venv.log')
        suffix='lib/python3.11/site-packages/audited_external_dependencies.pth'
        pth=(old/'agenticgen/.venv'/suffix).read_bytes(); (venv/suffix).write_bytes(pth)
        env=os.environ.copy(); env.pop('PYTHONPATH',None)
        env.update(AGENTICGEN_REPOSITORY=str(new),AGENTICGEN_PYTHON=str(venv/'bin/python'))
        env['LD_LIBRARY_PATH']='/opt/miniforge3/envs/molsteer-flowr-dtk/lib:'+env.get('LD_LIBRARY_PATH','')
        run([str(venv/'bin/python'),'-m','pip','install','--no-deps','--no-build-isolation',str(new/'agenticgen')],'install.log',env)
        run(['bash',str(new/'agenticgen/scripts/run.sh'),'doctor'],'doctor.log',env)
        if {str(p):sha(p) for p in protected} != before: raise ValueError('Protected science artifacts changed')
        receipt={'commit':revision,'repository':str(new),'time':time.time(),'phase':'installed',
            'source':'GitHub fetch and fixed detached checkout','preserved_sha256':before,
            'external_dependency_pth_sha256':hashlib.sha256(pth).hexdigest(),'science_started':False}
        (dep/'installation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'phase':'installed','commit':revision,'repository':str(new)}))


if __name__ == '__main__': main()
