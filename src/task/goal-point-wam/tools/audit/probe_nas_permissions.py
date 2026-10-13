#!/usr/bin/env python3
"""Compare Linux local/NFS creation permissions using only new diagnostic files.

This CPU file-I/O probe never modifies an existing file. FORCE_SYNC checks
server-refreshed attributes; a successful probe does not exclude old failures.
"""
import argparse
import pathlib,os,ctypes,struct,json,time
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--nas-output',type=pathlib.Path,required=True,help='New owned directory on the affected mount')
parser.add_argument('--local-output',type=pathlib.Path,required=True,help='New owned directory on local disk')
parser.add_argument('--observe-seconds',type=int,default=70)
args=parser.parse_args()
if not 0<=args.observe_seconds<=600:parser.error('observation must be bounded to0--600seconds')
out=args.nas_output;home=args.local_output
if out.exists() or home.exists():raise FileExistsError('Use new diagnostic directories; existing artifacts stay untouched')
if out.resolve()==home.resolve():raise ValueError('Local and NAS output paths must differ')
if not out.parent.is_dir() or not home.parent.is_dir():raise FileNotFoundError('Output parents must already exist')
out.mkdir();home.mkdir()
libc=ctypes.CDLL(None,use_errno=True);libc.statx.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_uint,ctypes.c_void_p];libc.statx.restype=ctypes.c_int

def fresh(path):
 buf=ctypes.create_string_buffer(256)
 if libc.statx(-100,os.fsencode(path),0x2100,0x7ff,buf)!=0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
 uid,gid=struct.unpack_from('=II',buf.raw,20);mode=struct.unpack_from('=H',buf.raw,28)[0];mask=struct.unpack_from('=I',buf.raw,0)[0]
 assert mask&2
 return {'mode':oct(mode&0o777),'uid':uid,'gid':gid,'inode':struct.unpack_from('=Q',buf.raw,32)[0],'ctime':struct.unpack_from('=qI',buf.raw,96),'mtime':struct.unpack_from('=qI',buf.raw,112)}
def access(path):
 try:
  with path.open('rb') as file:assert file.read()==b'permission probe\n'
  return True
 except OSError as e:return repr(e)
def sample(path):
 st=path.stat();cached={'mode':oct(st.st_mode&0o777),'uid':st.st_uid,'gid':st.st_gid}
 synced=fresh(path)
 return {'cached':cached,'force_sync':synced,'readable':access(path)}
report={'pid':os.getpid(),'uid':os.getuid(),'gid':os.getgid(),'locations':[str(out),str(home)],'events':[]};paths=[]
mask=os.umask(0o002)
for folder in [out,home]:
 for method in ['plain','exclusive','chmod_after_close','fchmod_before_close','plain_no_fsync']:
  path=folder/(method+'.bin');flags=os.O_CREAT|os.O_WRONLY|(os.O_EXCL if method=='exclusive' else os.O_TRUNC)
  fd=os.open(path,flags,0o666);os.write(fd,b'permission probe\n')
  st=os.fstat(fd);before={'mode':oct(st.st_mode&0o777),'uid':st.st_uid,'gid':st.st_gid}
  if method=='fchmod_before_close':os.fchmod(fd,0o644)
  if method!='plain_no_fsync':os.fsync(fd)
  os.close(fd)
  if method=='chmod_after_close':path.chmod(0o644)
  event={'location':'nas' if folder==out else 'local','method':method,'before_close':before,'after_close':sample(path)};report['events'].append(event);paths.append(path);print(json.dumps(event),flush=True)
os.umask(mask)
for delay in ([min(5,args.observe_seconds),max(0,args.observe_seconds-5)] if args.observe_seconds else []):
 time.sleep(delay)
 event={'delay_seconds':delay,'observations':[{'path':str(p),'sample':sample(p)} for p in paths]};report['events'].append(event);print(json.dumps(event),flush=True)
p=out/'result.json';p.write_text(json.dumps(report,indent=2)+'\n');p.chmod(0o644)
print('result',str(p),flush=True)
if any(event.get('after_close',{}).get('readable',True) is not True for event in report['events']):raise SystemExit(3)
if any(row['sample']['readable'] is not True for event in report['events'] for row in event.get('observations',[])):raise SystemExit(3)
