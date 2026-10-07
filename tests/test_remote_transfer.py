import io
from types import SimpleNamespace
import pytest
from evomolsteer.storage.remote_transfer import download_resumable

class Stream(io.BytesIO):
    def __init__(self,data,drop=False):super().__init__(data);self.drop=drop;self.calls=0
    def read(self,n):
        self.calls+=1
        if self.drop and self.calls>2:raise OSError('simulated connection drop')
        return super().read(n)

class Sftp:
    def __init__(self,data,drop=False):self.data=data;self.drop=drop;self.opens=0
    def stat(self,path):return SimpleNamespace(st_size=len(self.data))
    def open(self,path,mode):self.opens+=1;return Stream(self.data,self.drop)

def test_connection_drop_preserves_bytes_and_resumes_without_duplication(tmp_path):
    data=bytes(range(256))*10;p=tmp_path/'archive';source=Sftp(data,True)
    with pytest.raises(OSError):download_resumable(source,'remote',p,block_size=256)
    assert p.read_bytes()==data[:512]
    result=download_resumable(Sftp(data),'remote',p,block_size=256)
    assert p.read_bytes()==data and result['resumed_bytes']==512 and not result['checksum_verified']

def test_complete_file_still_requires_integrity_verification(tmp_path):
    p=tmp_path/'archive';p.write_bytes(b'abc');source=Sftp(b'abc')
    result=download_resumable(source,'remote',p)
    assert source.opens==0 and result['checksum_verified'] is False

def test_inconsistent_partial_size_is_not_silently_overwritten(tmp_path):
    p=tmp_path/'archive';p.write_bytes(b'excess')
    with pytest.raises(ValueError):download_resumable(Sftp(b'abc'),'remote',p)
    assert p.read_bytes()==b'excess'
