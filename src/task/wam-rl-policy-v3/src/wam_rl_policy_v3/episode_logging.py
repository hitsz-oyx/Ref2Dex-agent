"""Bounded per-rank episode evidence; aggregate counters continue after the cap."""
import json
from pathlib import Path


class EpisodeLog:
    def __init__(self,path,max_bytes=8*1024*1024):
        self.path=Path(path);self.max_bytes=int(max_bytes);self.bytes=0;self.written=0;self.dropped=0
        if self.max_bytes<=0:raise ValueError('positive episode log budget required')
        with self.path.open('x'):pass

    def append(self,result):
        line=json.dumps(result,allow_nan=False,separators=(',',':'))+'\n';size=len(line.encode('utf-8'))
        if self.bytes+size>self.max_bytes:self.dropped+=1;return
        with self.path.open('a') as stream:stream.write(line)
        self.bytes+=size;self.written+=1

    def summary(self):
        return dict(file=str(self.path),max_bytes=self.max_bytes,bytes=self.bytes,written=self.written,dropped_after_cap=self.dropped)
