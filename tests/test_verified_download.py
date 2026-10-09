from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import hashlib
import threading

import pytest

from evomolsteer.storage.download import download_verified


@pytest.fixture
def archive_server():
    data=bytes(range(256))*100;requests=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            lo,hi=map(int,self.headers['Range'][6:].split('-'));requests.append((lo,hi))
            self.send_response(206);self.send_header('Content-Range',f'bytes {lo}-{hi}/{len(data)}')
            self.send_header('Content-Length',str(hi-lo+1));self.end_headers();self.wfile.write(data[lo:hi+1])
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}/archive',data,requests
    server.shutdown();server.server_close();thread.join()


def test_parallel_download_matches_source_and_verified_reuse(tmp_path,archive_server):
    url,data,requests=archive_server;path=tmp_path/'archive.tar.gz';md5=hashlib.md5(data).hexdigest()
    result=download_verified(url,path,size=len(data),md5=md5,workers=4,chunk_size=3000)
    assert path.read_bytes()==data and result['verified'] and not result['reused']
    assert len(requests)==9 and not path.with_name(path.name+'.ranges').exists()
    download_verified(url,path,size=len(data),md5=md5,workers=4,chunk_size=3000)
    assert len(requests)==9


def test_bad_checksum_never_publishes_archive(tmp_path,archive_server):
    url,data,_=archive_server;path=tmp_path/'bad.tar.gz'
    with pytest.raises(ValueError,match='published checksum'):
        download_verified(url,path,size=len(data),md5='0'*32,workers=3,chunk_size=10000)
    assert not path.exists()
    assert path.with_name(path.name+'.ranges').is_dir()
