import importlib.util,json,hashlib,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dispatch_path_round import verify_retention

def test_report_checksum_and_path_safety(tmp_path):
    source=tmp_path/'score.json';source.write_bytes(b'{"score":8}')
    report=tmp_path/'retention.json'
    r={'status':'complete','files':[{'path':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}]}
    report.write_text(json.dumps(r));assert verify_retention(report)['status']=='complete'
    source.write_bytes(b'{"score":9}')
    with pytest.raises(ValueError):verify_retention(report)
    r['files'][0]['path']='../escaped.json';report.write_text(json.dumps(r))
    with pytest.raises(ValueError):verify_retention(report)
def test_incomplete_report_rejected(tmp_path):
    path=tmp_path/'report.json';path.write_text('{"status":"running"}')
    with pytest.raises(ValueError):verify_retention(path)
