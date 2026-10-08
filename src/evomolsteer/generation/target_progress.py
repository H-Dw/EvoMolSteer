"""Empty per-target claims and terminal markers shared by campaign workers."""
from dataclasses import dataclass
import os
from pathlib import Path
import uuid

from ..storage.transactions import atomic_json,checked_path

FORMAT='evomolsteer.target_progress.v1'
SUFFIXES=('running','finished','error')


class TargetClaimConflict(RuntimeError):
    """A competing completion or a changed claim prevents normal publication."""


def _atomic_text(path,text):
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8',newline='\n') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


@dataclass(frozen=True)
class TargetClaim:
    key:str
    token:str
    identity:tuple[int,int]


class TargetProgress:
    def __init__(self,output,catalog):
        self.root=Path(output)/'generation_progress'
        checked_path(self.root,output)
        self.root.mkdir(parents=True,exist_ok=True)
        self.keys={row['target_id']:row['key'] for row in catalog}
        self.valid_keys=set(self.keys.values())
        if len(self.keys)!=len(catalog) or len(self.valid_keys)!=len(catalog) or any(
            not isinstance(key,str) or not key or key in ('.','..') or any(c in key for c in '/\\:')
            for key in self.valid_keys):raise ValueError('Distinct, safe single-filename target keys required')

    def write_index(self):
        atomic_json(self.root/'targets.json',{'format':FORMAT,'targets':self.keys,
                    'filename':'<target_key>.running|finished|error',
                    'claim_rule':'Atomic exclusive creation; never steal or automatically retry a marker'})

    def path(self,key,suffix):
        if key not in self.valid_keys or suffix not in SUFFIXES:raise ValueError('Unknown target marker')
        return self.root/(key+'.'+suffix)

    def statuses(self,key):
        return [suffix for suffix in SUFFIXES if os.path.lexists(self.path(key,suffix))]

    def counts(self):
        counts={suffix:0 for suffix in (*SUFFIXES,'available')}
        for key in self.valid_keys:
            present=self.statuses(key)
            # An error takes precedence in a malformed/external mixed state.
            status=next((s for s in ('error','finished','running') if s in present),'available')
            counts[status]+=1
        return counts

    def _owned(self,claim):
        path=self.path(claim.key,'running')
        if not path.is_file() or path.is_symlink():return False
        try:stat=path.stat(follow_symlinks=False)
        except FileNotFoundError:return False
        return (stat.st_dev,stat.st_ino)==claim.identity

    def claim(self,key):
        """Create the empty running marker before any target job is prepared.

        O_EXCL arbitrates simultaneous callers; the second terminal-marker
        check also closes a check/create race during an existing completion.
        """
        if self.statuses(key):return None
        running=self.path(key,'running')
        try:fd=os.open(running,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
        except FileExistsError:return None
        try:
            os.fsync(fd);stat=os.fstat(fd)
        finally:os.close(fd)
        claim=TargetClaim(key,uuid.uuid4().hex,(stat.st_dev,stat.st_ino))
        if any(os.path.lexists(self.path(key,s)) for s in ('finished','error')):
            if self._owned(claim):running.unlink()
            return None
        return claim

    def finish(self,claim):
        """Publish finished without ever overwriting a competing finished file.

        A same-directory hard link followed by unlink is a no-replace rename
        of the empty claim. Both names briefly mean occupied, never available.
        """
        if 'finished' in self.statuses(claim.key):
            reason='Concurrent target execution: a finished marker already exists when this worker completes'
            self.fail(claim,reason+'\n')
            raise TargetClaimConflict(reason)
        if 'error' in self.statuses(claim.key) or not self._owned(claim):
            reason='Concurrent target execution or changed ownership: the running claim is missing/replaced or an error marker exists'
            self.fail(claim,reason+'\n')
            raise TargetClaimConflict(reason)
        try:os.link(self.path(claim.key,'running'),self.path(claim.key,'finished'))
        except FileExistsError:
            reason='Concurrent target execution: another worker published finished during completion'
            self.fail(claim,reason+'\n')
            raise TargetClaimConflict(reason)
        self.path(claim.key,'running').unlink()

    def fail(self,claim,detail):
        """Convert a competing finished to error, preserving all diagnostics."""
        error=self.path(claim.key,'error');previous=''
        if error.is_file() and not error.is_symlink():
            previous=error.read_text(encoding='utf-8',errors='replace')
        finished=self.path(claim.key,'finished')
        if os.path.lexists(finished):
            if not finished.is_file() or finished.is_symlink():
                detail+='\nThe competing finished marker is not a regular file; it was left untouched.\n'
            else:os.replace(finished,error)
        elif self._owned(claim):os.replace(self.path(claim.key,'running'),error)
        _atomic_text(error,previous+('\n--- additional error ---\n' if previous else '')+detail)
        if self._owned(claim):self.path(claim.key,'running').unlink()

    def migrate(self,key,status,detail=''):
        """Backfill markers once for a legacy campaign, without starting work."""
        if self.statuses(key):return
        if status=='finished':
            with self.path(key,status).open('xb'):pass
        elif status=='running':
            with self.path(key,status).open('xb'):pass
        elif status=='error':_atomic_text(self.path(key,status),detail or 'Legacy campaign reported a target failure.\n')
