"""A real interrupted HTTP stream must resume after acknowledged bytes."""
import importlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import requests

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))


def test_interrupted_range_retains_valid_prefix(tmp_path,monkeypatch):
    module=importlib.import_module('fetch_taco_parallel')
    payload=bytes(range(251))*(3*2**20//251+1);payload=payload[:3*2**20];seen=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            lo,hi=map(int,self.headers['Range'].split('=',1)[1].split('-'));seen.append(lo)
            self.send_response(206);self.send_header('Content-Length',str(hi-lo+1))
            self.send_header('Content-Range',f'bytes {lo}-{hi}/{len(payload)}');self.end_headers()
            # Every request starting at zero dies midstream; restarting from zero cannot finish.
            if lo==0:self.wfile.write(payload[:1536*1024]);self.wfile.flush();self.close_connection=True
            else:self.wfile.write(payload[lo:hi+1])
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    original=requests.Session
    def session():
        # Both downloader routes remain on the local fault server for this test.
        class DirectSession(original):
            @property
            def proxies(self):return {}
            @proxies.setter
            def proxies(self,value):pass
        return DirectSession()
    monkeypatch.setattr(module.requests,'Session',session)
    monkeypatch.setattr(module,'BASE',f'http://127.0.0.1:{server.server_port}/');monkeypatch.setattr(module,'SUFFIX','')
    monkeypatch.setattr(module,'ARCHIVES',dict(hand=('archive.zip',len(payload)),poses=('',1),meshes=('',1)))
    (tmp_path/'archives').mkdir();(tmp_path/'acquisition-status.json').write_text(json.dumps(dict(status='PARTIAL',pid=999999999)))
    monkeypatch.setattr(sys,'argv',['fetch_taco_parallel','--output',str(tmp_path),'--workers','1','--seconds','60'])
    try:module.main()
    finally:server.shutdown();server.server_close();thread.join()
    target=tmp_path/'archives/Hand_Poses.zip'
    assert target.exists(), 'restarting the same failed range loses its valid prefix'
    assert target.read_bytes()==payload
    assert seen==[0,1024*1024]
