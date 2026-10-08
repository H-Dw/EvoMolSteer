"""Short, process-safe campaign transactions; never hold this lock in inference."""
from contextlib import contextmanager
from pathlib import Path

from ..io import read_json
from ..storage.transactions import atomic_json,exclusive_lock,checked_path
from .target_progress import FORMAT as PROGRESS_FORMAT


class CampaignJournal:
    def __init__(self,root,configuration,catalog,progress,format):
        self.root=Path(root);self.progress=progress;self.format=format
        self.path=self.root/'campaign_state.json';self.lock=self.root/'.target_campaign.lock'
        with exclusive_lock(self.lock,blocking=True):
            if self.path.exists():
                state=read_json(self.path)
                if state.get('format')!=format or state['config']!=configuration or state['catalog']!=catalog:
                    raise ValueError('Resume configuration or target input hashes changed; use a new output collection')
                if state.get('progress_format') not in (None,PROGRESS_FORMAT):raise ValueError('Unsupported progress markers')
                if 'progress_format' not in state:
                    for row in catalog:
                        record=state['targets'][row['target_id']]
                        suffix={'complete':'finished','archived':'finished','failed':'error','running':'running'}.get(record['state'])
                        if suffix:
                            detail=''
                            if suffix=='error':
                                detail='Migrated legacy target failure: '+record.get('error','')+'\n'
                                for attempt in record.get('attempts',[]):
                                    log=checked_path(self.root/attempt['log'],self.root)
                                    if log.is_file():detail+='\n--- legacy generation log ---\n'+log.read_text(encoding='utf-8',errors='replace')
                            progress.migrate(row['key'],suffix,detail)
            else:
                allowed={self.lock.name,progress.root.name}
                if any(p.name not in allowed for p in self.root.iterdir()):
                    raise FileExistsError('Collection output is not owned by this controller')
                state={'format':format,'config':configuration,'catalog':catalog,'status':'pending','archives':[],
                       'targets':{r['target_id']:{'state':'pending','attempts':[]} for r in catalog}}
                atomic_json(self.root/'targets_resolved.json',{'format':'evomolsteer.target_collection.v1',
                            'dataset':configuration['input_dataset'],'targets':catalog})
            state['progress_format']=PROGRESS_FORMAT
            progress.write_index();self._save(state)

    def _save(self,state):
        counts=self.progress.counts();records=state['targets'].values()
        if counts['running']:state['status']='running'
        elif state.get('archive_error'):state['status']='archive_error'
        elif state.get('archive_pending'):state['status']='archiving'
        elif all(r['state'] in ('complete','archived') for r in records) and not counts['error']:state['status']='complete'
        elif counts['available']:state['status']='pending'
        elif counts['error']:state['status']='partial_failure'
        else:state['status']='markers_only'
        totals={'targets':len(state['targets']),
                'successful_targets':sum(r['state'] in ('complete','archived') for r in state['targets'].values()),
                'failed_targets':counts['error'],'archive_groups':len(state['archives']),
                'archive_bytes':sum(a['metadata']['archive_bytes'] for a in state['archives'])}
        atomic_json(self.path,state)
        atomic_json(self.root/'results_summary.json',{'format':self.format,'status':state['status'],'totals':totals,
                    'progress':counts,'generation_progress':self.progress.root.name,
                    'targets':{key:{k:v for k,v in row.items() if k in ('state','summary','dataset','archive','attempts')}
                               for key,row in state['targets'].items()},'archives':state['archives'],
                    'archive_error':state.get('archive_error')})

    @contextmanager
    def transaction(self):
        with exclusive_lock(self.lock,blocking=True):
            state=read_json(self.path)
            yield state
            self._save(state)

    def snapshot(self):
        with exclusive_lock(self.lock,blocking=True):return read_json(self.path)

    def summary(self):
        with exclusive_lock(self.lock,blocking=True):return read_json(self.root/'results_summary.json')
