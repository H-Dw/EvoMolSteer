"""Repeat selection statistics and assert deterministic artifacts."""
import argparse
import importlib
from pathlib import Path
from evomolsteer.io import digest,write_json,read_json
from evomolsteer.evidence import build_bundle

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    root=Path(p.parse_args().output)
    paths=[f for f in sorted((root/'discovery').rglob('*')) if f.is_file()]
    paths += [root/'agents/evidence_bundle.json']
    before={str(f.relative_to(root)):digest(f) for f in paths}
    for name in ['enrichment','differential','trends','pca']:
        print('Repeating',name,flush=True)
        importlib.import_module('evomolsteer.'+name).run(root)
    build_bundle(root)
    changes=[name for name,sha in before.items() if digest(root/name)!=sha]
    residual=read_json(root/'discovery/trends/event_diagnostics.json')['max_selection_noise_identity_residual']
    assert residual<1e-10
    write_json(root/'verification/determinism.json',{'checked_files':len(before),'changed_files':changes,
        'max_selection_noise_identity_residual':residual,'passed':not changes})
    assert not changes,changes
    print('Deterministic files:',len(before))
