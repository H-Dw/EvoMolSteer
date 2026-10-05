"""Verify deployed reward content, accounting explicitly for Git line endings."""
import argparse
import hashlib
import json
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def audit(configs,datasets,output):
    records=[]
    for folder in sorted(Path(datasets).glob('round_*')):
        if not folder.is_dir():continue
        deployed=folder/'results'/folder.name/'reward_program.json'
        local=Path(configs)/(folder.name+'.json')
        a,b=read_json(local),read_json(deployed)
        if a!=b:raise ValueError('Deployed reward differs from local frozen content: '+folder.name)
        records.append({'campaign':folder.name,'json_content_identical':True,'canonical_json_sha256':canonical_hash(a),
                        'local_file_sha256':digest(local),'deployed_file_sha256':digest(deployed),
                        'byte_identical':local.read_bytes()==deployed.read_bytes(),
                        'identical_after_LF_normalization':local.read_bytes().replace(b'\r\n',b'\n')==deployed.read_bytes().replace(b'\r\n',b'\n')})
    result={'status':'passed','note':'Local Windows and deployed Linux file hashes can differ because Git normalizes CRLF to LF. JSON equality is checked, and both byte hashes are retained.','records':records}
    write_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configs',required=True);p.add_argument('--datasets',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=audit(a.configs,a.datasets,a.output);print('Verified deployed JSON content for',len(r['records']),'rounds')
