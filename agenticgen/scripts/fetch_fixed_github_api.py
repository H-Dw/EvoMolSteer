"""Fetch hash-verified GitHub objects when the Git HTTPS proxy returns HTTP 503.

Uses the same inherited proxy environment; no alternate repository or source upload.
Only installs immutable objects into the existing Git database, never scientific data.
"""
import argparse
import base64
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

OLD='/opt/MolSteer/runtimes/agenticgen_d1163b7e'
API='https://api.github.com/repos/H-Dw/EvoMolSteer/git/'
PREFIXES=('agenticgen/','src/evomolsteer/generation/','src/evomolsteer/adaptive_schedule/vendor/')


def object_id(kind,raw):
    return hashlib.sha1((kind+' '+str(len(raw))+'\0').encode()+raw).hexdigest()


def tree_bytes(entries):
    entries=sorted(entries,key=lambda e:(e['name']+('/' if e['type']=='tree' else '')).encode())
    return b''.join((e['mode'].lstrip('0')+' '+e['name']).encode()+b'\0'+bytes.fromhex(e['sha']) for e in entries)


def commit_bytes(document,target):
    if document['sha']!=target: raise ValueError('GitHub returned another commit')
    headers=['tree '+document['tree']['sha']]+['parent '+p['sha'] for p in document['parents']]
    def identity(role,tz):
        v=document[role];stamp=int(datetime.datetime.fromisoformat(v['date'].replace('Z','+00:00')).timestamp())
        return f"{role} {v['name']} <{v['email']}> {stamp} {tz}"
    # API timestamps normalize the original zone; accept reconstruction only if
    # its complete Git object digest matches the requested immutable commit.
    zones=['+0800','+0000']+[f'{s}{h:02d}{m:02d}' for s in ('+','-') for h in range(15) for m in (0,15,30,45)]
    for az in zones:
        for cz in zones:
            for msg in (document['message'],document['message'].rstrip('\n')+'\n'):
                raw=('\n'.join(headers+[identity('author',az),identity('committer',cz)])+'\n\n'+msg).encode()
                if object_id('commit',raw)==target:return raw
    raise ValueError('Unsupported commit headers; refusing an approximate reconstruction')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True);args=parser.parse_args()
    target=args.commit
    if not re.fullmatch('[0-9a-f]{40}',target):raise ValueError('Full fixed commit required')
    def git(*args,data=None):return subprocess.check_output(['git','-C',OLD,*args],input=data)
    known=set(git('cat-file','--batch-all-objects','--batch-check=%(objectname)').decode().splitlines())
    records=[]
    def api(suffix):
        req=urllib.request.Request(API+suffix,headers={'User-Agent':'agenticgen-fixed-commit-deployment'})
        with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
    def put(kind,raw,expected):
        if object_id(kind,raw)!=expected:raise ValueError('GitHub object digest mismatch')
        if git('hash-object','-w','-t',kind,'--stdin',data=raw).decode().strip()!=expected:
            raise ValueError('Git object write differs')
        records.append({'type':kind,'oid':expected,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    commits=0
    def ensure_commit(oid):
        nonlocal commits
        if oid in known:return
        commits+=1
        if commits>32:raise ValueError('Commit ancestry exceeds bounded repair fetch')
        d=api('commits/'+oid);put('commit',commit_bytes(d,oid),oid);known.add(oid)
        for parent in d['parents']:ensure_commit(parent['sha'])
    ensure_commit(target)
    root=api('commits/'+target)['tree']['sha'];tree=api('trees/'+root+'?recursive=1')
    if tree['sha']!=root or tree.get('truncated') is not False:raise ValueError('Incomplete GitHub tree')
    children={'':[]};expected={'':root}
    for item in tree['tree']:
        parent,_,name=item['path'].rpartition('/')
        children.setdefault(parent,[]).append({**item,'name':name})
        if item['type']=='tree':children.setdefault(item['path'],[]);expected[item['path']]=item['sha']
    for path,entries in children.items():
        if expected[path] not in known:put('tree',tree_bytes(entries),expected[path]);known.add(expected[path])
    missing={x['sha'] for x in tree['tree'] if x['type']=='blob' and x['path'].startswith(PREFIXES) and x['sha'] not in known}
    def blob(oid):
        d=api('blobs/'+oid)
        if d['sha']!=oid or d['encoding']!='base64':raise ValueError('Unexpected GitHub blob encoding')
        put('blob',base64.b64decode(d['content']),oid)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(blob,sorted(missing)))
    folder=Path('/opt/MolSteer/operations')/('agenticgen_deploy_'+target[:8]);folder.mkdir(parents=True,exist_ok=True)
    (folder/'github_object_receipt.json').write_text(json.dumps({'commit':target,'root_tree':root,'source':API,
        'objects':sorted(records,key=lambda x:x['oid']),'existing_objects_reused':True,'source_upload':False},indent=2)+'\n')
    print('AGENTICGEN_EVENT '+json.dumps({'phase':'github_objects_verified','commit':target,'downloaded_blobs':len(missing)}),flush=True)


if __name__=='__main__':main()
