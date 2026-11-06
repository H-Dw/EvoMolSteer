"""Suixiang image transport using the nature-figure prompt helper.

The image provider requires multipart edits; authentication is environment-only.
Submission is never automatically retried. Polling does not resubmit a task.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import uuid


def write_json(path,value):
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def request_json(url,key,data=None,content_type=None):
    headers={'Authorization':'Bearer '+key}
    if content_type:headers['Content-Type']=content_type
    req=urllib.request.Request(url,data=data,headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=60) as r:return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body=e.read().decode('utf-8',errors='replace').replace(key,'[REDACTED]')
        raise RuntimeError('HTTP '+str(e.code)+': '+body[:600]) from None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['submit','poll'])
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--skill-script')
    ap.add_argument('--reference-image',action='append')
    args=ap.parse_args();out=Path(args.outdir)
    key=os.environ.get('SUIXIANG_API_KEY')
    if not key:raise SystemExit('SUIXIANG_API_KEY is absent from the process environment')
    base='https://sui-xiang.com';model='gpt-image-2.6-sunburst'
    if args.mode=='submit':
        if (out/'submission.json').exists():raise SystemExit('Existing submission: poll it instead of resubmitting')
        spec=importlib.util.spec_from_file_location('nature_figure_image_helper',args.skill_script)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        prompt=helper.build_prompt(argparse.Namespace(prompt=None,prompt_file=str(out/'prompt.md'),raw=True))
        fields={'model':model,'prompt':prompt,'size':'3072x1536','quality':'high','complexity':'high','n':'1'}
        refs=([Path(p) for p in args.reference_image] if args.reference_image else
              [out/'molecular_reference.png',out/'atom_graph.png'])
        metadata={'base_url':base,'endpoint':'/v1/images/edits/async','requested_model':model,
            'fields':fields,'references':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in refs],
            'authentication':'SUIXIANG_API_KEY environment variable; credential never persisted',
            'prompt_helper':'nature-figure generate_openrouter_schematic.py build_prompt; vendor-specific multipart transport'}
        write_json(out/'request_metadata.json',metadata)
        boundary='figure-'+uuid.uuid4().hex;parts=[]
        for name,value in fields.items():
            parts.append(('--'+boundary+'\r\nContent-Disposition: form-data; name="'+name+'"\r\n\r\n'+value+'\r\n').encode())
        for p in refs:
            parts.extend([('--'+boundary+'\r\nContent-Disposition: form-data; name="image[]"; filename="'+p.name+'"\r\nContent-Type: image/png\r\n\r\n').encode(),p.read_bytes(),b'\r\n'])
        parts.append(('--'+boundary+'--\r\n').encode())
        write_json(out/'submission_claim.json',{'requested_model':model,'automatic_retry':False})
        try:d=request_json(base+'/v1/images/edits/async',key,b''.join(parts),'multipart/form-data; boundary='+boundary)
        except Exception as e:
            write_json(out/'submission_failure.json',{'error_type':type(e).__name__,'message':str(e).replace(key,'[REDACTED]')})
            raise SystemExit(str(e).replace(key,'[REDACTED]')) from None
        write_json(out/'submission.json',{k:v for k,v in d.items() if k in ['id','status','model','created_at']})
        if not d.get('id'):raise SystemExit('No task id returned; inspect the preserved submission metadata')
        print(json.dumps({'status':d.get('status'),'task_id':d['id'],'requested_model':model}));return
    submitted=json.loads((out/'submission.json').read_text(encoding='utf-8'))
    d=request_json(base+'/v1/images/tasks/'+submitted['id'],key)
    state={k:d[k] for k in ['id','status','model','created_at','completed_at'] if k in d}
    write_json(out/'task_status.json',state)
    if d.get('status')=='failed':
        err=str(d.get('error','Unknown generation failure')).replace(key,'[REDACTED]')
        write_json(out/'generation_failure.json',dict(state,error=err))
        print(json.dumps(dict(state,error=err)));return
    if d.get('status')=='completed':
        url=d.get('image_url')
        if not url:
            rows=d.get('result',{}).get('data',[]);url=rows[0].get('url') if rows else None
        if not url:raise SystemExit('Completed task has no image URL')
        dest=out/'scale_calibration_schematic.png'
        if not dest.exists():
            with urllib.request.urlopen(url,timeout=60) as r:content=r.read()
            dest.write_bytes(content)
        write_json(out/'generation_receipt.json',dict(state,requested_model=model,
            output_file=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            usage=d.get('result',{}).get('usage'),original_api_image_unmodified=True))
        state['saved_file']=dest.name
    print(json.dumps(state))


if __name__=='__main__':main()
