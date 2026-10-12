"""ZIP extraction must coalesce tiny local-header requests without changing bytes."""
import io
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))
from launch_egodex_full_pipeline import IndexedRemoteZip, packet_groups


class Response:
    def __init__(self,payload,lo=0,hi=None,head=False):
        self.status_code=200 if head else 206
        self.headers={'Content-Length':str(len(payload)),'ETag':'pinned'}
        if not head:self.headers['Content-Range']='bytes %d-%d/%d'%(lo,hi,len(payload))
        self.data=payload[lo:hi+1] if not head else b''
    def raise_for_status(self):pass
    def iter_content(self,n):yield self.data
    def close(self):pass


class Session:
    def __init__(self,payload):self.payload=payload;self.calls=[]
    def head(self,*args,**kwargs):return Response(self.payload,head=True)
    def get(self,*args,**kwargs):
        lo,hi=map(int,kwargs['headers']['Range'].split('=')[1].split('-'))
        self.calls.append((lo,hi));return Response(self.payload,lo,hi)


def fixture():
    buf=io.BytesIO();data={'part2/a/0.hdf5':b'annotations'*10000,'part2/a/0.mp4':b'video'*30000}
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,payload in data.items():archive.writestr(name,payload)
    payload=buf.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:start=z.start_dir
    session=Session(payload)
    handle=IndexedRemoteZip('https://fake',session,1000000,1e20,start,payload[start:])
    return data,handle,session


def test_pair_payloads_and_crc_unchanged_with_one_request_instead_of_six():
    expected,handle,session=fixture()
    with zipfile.ZipFile(handle) as z:
        assert not session.calls, 'central directory is already cached'
        for name,payload in expected.items():assert z.read(name)==payload
    baseline=len(session.calls)
    assert baseline>=6, 'reproduces tiny ZIP local-header/name/body request overhead'
    expected,handle,session=fixture()
    with zipfile.ZipFile(handle) as z:
        assert handle.prefetch_members([z.getinfo(name) for name in expected])
        for name,payload in expected.items():assert z.read(name)==payload
    assert len(session.calls)==1


def test_prefetch_budget_falls_back_without_an_extra_request_or_corrupting_seek():
    expected,handle,session=fixture()
    with zipfile.ZipFile(handle) as z:
        assert not handle.prefetch_members([z.getinfo(name) for name in expected],max_bytes=1)
        assert not session.calls
        for name,payload in expected.items():assert z.read(name)==payload
    assert len(session.calls)>=6


def test_multiple_pairs_share_one_range_in_physical_order():
    buf=io.BytesIO();expected={}
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for episode in [0,2,10]:
            for suffix in ['hdf5','mp4']:
                name='part2/a/%d.%s'%(episode,suffix)
                expected[name]=('%s payload'%name).encode()*100
                z.writestr(name,expected[name])
    payload=buf.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        start=z.start_dir;infos={i.filename:i for i in z.infolist()}
    packets=[(n,[n,n[:-5]+'.mp4'],True) for n in sorted(infos) if n.endswith('.hdf5')]
    groups=list(packet_groups(packets,infos,1024*1024))
    assert len(groups)==1
    assert [p[0] for p in groups[0]]==['part2/a/0.hdf5','part2/a/2.hdf5','part2/a/10.hdf5']
    session=Session(payload)
    handle=IndexedRemoteZip('https://fake',session,1000000,1e20,start,payload[start:])
    with zipfile.ZipFile(handle) as z:
        assert handle.prefetch_members([infos[n] for p in groups[0] for n in p[1]])
        for p in groups[0]:
            for n in p[1]:assert z.read(n)==expected[n]
    assert len(session.calls)==1


def test_large_pair_stays_intact_for_bounded_fallback():
    expected,handle,session=fixture()
    with zipfile.ZipFile(handle) as z:
        infos={i.filename:i for i in z.infolist()}
        pair=('part2/a/0.hdf5',list(expected),True)
        groups=list(packet_groups([pair],infos,1))
        assert groups==[[pair]]
        assert not handle.prefetch_members([infos[n] for n in pair[1]],max_bytes=1)
        for n in pair[1]:assert z.read(n)==expected[n]
