"""Read-only progress overview; no signed URLs or model execution."""
import json
from pathlib import Path
import time

REPO=Path(__file__).resolve().parents[5]
BASE=REPO/'outputs/3d-data-preparation'
NAMES=['hot3d-native-full-20261014-r1','hot3d-posemesh-full-20261014-r1',
       'taco-native-full-20261014-r1','taco-posemesh-full-20261014-r1','unidexgrasp-expansion-20261014-r1',
       'hocap-posemesh-reuse-20261014-r1','dexycb-posemesh-reuse-20261014-r1','native3d-live-corpus-20261014-r1']


for name in NAMES:
    root=BASE/name
    file=next((root/f for f in ['acquisition-status.json','progress.json','conversion-audit.json'] if (root/f).exists()),None)
    if not file:print(name,'NOT_STARTED');continue
    raw=json.loads(file.read_text())
    keys=['status','pid','completed_files','total_files','completed_bytes','total_bytes','network_bytes',
          'converted_sequences','converted_frames','completed_batches','total_batches','active_batch',
          'sequences','frames','objects','failed_sequences','skipped_sequences','source_counts']
    summary={k:raw[k] for k in keys if k in raw}
    if 'pid' in raw:summary['process_alive']=(Path('/proc')/str(raw['pid'])).exists()
    if 'checked_at' in raw:summary['status_age_seconds']=round(time.time()-raw['checked_at'],1)
    rate=raw.get('payload_bytes_per_s',0)
    if rate>0 and raw['status']=='RUNNING':
        remaining=max(0,raw['total_bytes']-raw['completed_bytes']-raw.get('active_received_bytes',0))
        summary.update(network_MB_per_s=round(rate/1e6,2),estimated_remaining_minutes=round(remaining/rate/60,1))
    print(name,json.dumps(summary,ensure_ascii=False))
