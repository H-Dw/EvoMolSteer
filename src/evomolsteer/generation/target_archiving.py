"""Serialize archive publication while other workers continue claiming targets."""
import json
from pathlib import Path
import traceback

from ..storage.transactions import atomic_json,exclusive_lock,WriterBusyError
from ..storage.target_archive import archive_target_directories,verify_target_archive,retire_archived_directories


class TargetArchiveCoordinator:
    def __init__(self,root,cfg,journal,progress):
        self.root=Path(root);self.cfg=cfg;self.journal=journal;self.progress=progress

    def _all_finished(self,ids):
        return all(self.progress.statuses(self.progress.keys[identifier])==['finished'] for identifier in ids)

    def _retire(self,item):
        if not self._all_finished(item['target_ids']):raise ValueError('A target marker changed; archive retirement is blocked')
        directories={k:self.root/p for k,p in item['directories'].items()}
        result=retire_archived_directories(self.root/item['path'],item['metadata'],directories,owned_root=self.root/'targets')
        with self.journal.transaction() as state:
            saved=next(a for a in state['archives'] if a['path']==item['path'])
            saved['retirement']=result

    def _reserve(self,flush):
        with self.journal.transaction() as state:
            if state.get('archive_pending'):return state['archive_pending']
            rows=[r for r in state['catalog'] if state['targets'][r['target_id']]['state']=='complete'
                  and self.progress.statuses(r['key'])==['finished']]
            counts=self.progress.counts();exhausted=not counts['available'] and not counts['running']
            if len(rows)>=self.cfg.archive_every:rows=rows[:self.cfg.archive_every]
            elif not rows or not (flush or exhausted):return None
            pending={'path':f"archives/targets_{len(state['archives']):04d}.tar.gz",
                     'target_ids':[r['target_id'] for r in rows],
                     'ids':{r['key']:r['target_id'] for r in rows},
                     'directories':{r['key']:state['targets'][r['target_id']]['dataset'] for r in rows}}
            # Freeze group membership before slow compression; other completions
            # must not change recovery membership after a process interruption.
            state['archive_pending']=pending
            return pending

    def _pack(self,pending):
        path=self.root/pending['path'];sidecar=Path(str(path)+'.json')
        directories={k:self.root/p for k,p in pending['directories'].items()}
        if not self._all_finished(pending['target_ids']):raise ValueError('Only finished target tasks can be archived')
        if path.exists():
            verified=verify_target_archive(path,sidecar if sidecar.exists() else None)
            if verified['manifest']['targets']!={k:{'target_id':v} for k,v in pending['ids'].items()}:
                raise ValueError('Published group differs from the reserved target list')
            verified.pop('manifest');metadata=verified
            if not sidecar.exists():atomic_json(sidecar,metadata)
        else:
            partial=Path(str(path)+'.partial')
            if partial.exists():
                n=0
                while Path(str(partial)+f'.abandoned_{n}').exists():n+=1
                partial.rename(Path(str(partial)+f'.abandoned_{n}'))
            metadata=archive_target_directories(directories,path,target_ids=pending['ids'],
                                              compression_level=self.cfg.compression_level)
        item={k:pending[k] for k in ('path','target_ids','directories')}
        item.update(metadata=metadata,retirement='pending' if self.cfg.remove_archived_targets else 'kept')
        with self.journal.transaction() as state:
            if state.get('archive_pending')!=pending or not self._all_finished(pending['target_ids']):
                raise ValueError('Archive reservation or target progress changed during compression')
            state['archives'].append(item)
            for identifier in pending['target_ids']:
                state['targets'][identifier].update(state='archived',archive=item['path'])
            del state['archive_pending'];state.pop('archive_error',None)
        # Locator/state are durable before retirement; no state lock in gzip,
        # verification or deletion. A separate OS lock excludes other archivers.
        if self.cfg.remove_archived_targets:self._retire(item)
        print(json.dumps({'event':'target_group_archived','target_count':len(pending['target_ids']),
                         'source_bytes':metadata['source_bytes'],'archive_bytes':metadata['archive_bytes'],
                         'reduction_percent':metadata['reduction_percent'],'path':str(path)}),flush=True)

    def run(self,*,flush=False):
        if not self.cfg.compress:return 'disabled'
        try:
            with exclusive_lock(self.root/'.target_archive.lock'):
                try:
                    for item in self.journal.snapshot()['archives']:
                        if item['retirement']=='pending':self._retire(item)
                    while (pending:=self._reserve(flush)) is not None:self._pack(pending)
                    with self.journal.transaction() as state:state.pop('archive_error',None)
                    return 'ready'
                except Exception:
                    detail=traceback.format_exc()
                    with self.journal.transaction() as state:state['archive_error']=detail
                    print(json.dumps({'event':'archive_error','error':detail}),flush=True)
                    return 'error'
        except WriterBusyError:
            return 'busy'  # never block another target's inference on gzip
