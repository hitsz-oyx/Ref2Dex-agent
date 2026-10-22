"""Summarize actual phase entry and forecast validity; preserve GT implementation failures."""
import argparse,collections,json
from pathlib import Path

def audit(run,mechanism_run=None):
    episodes=[]
    for stage in ('mechanism','comparators'):
        for path in sorted(((mechanism_run if stage=='mechanism' and mechanism_run is not None else run)/stage).glob('g*/manifest.json')):
            for row in json.loads(path.read_text())['runs']:
                infos=json.loads((path.parent/row['id']/'consumer.json').read_text());first={}
                for tick,info in enumerate(infos):first.setdefault(info['phase'],tick)
                episodes.append(dict(id=row['id'],subset=row['subset'],arm=row['arm'],sequence=row['sequence'],first_phase_tick=first,close_entered='close' in first or 'lift' in first or 'hold' in first,valid_controls=row['encoder_covered_controls'],controls=row['controls'],predicted_contact_controls=sum(i.get('predicted_contact_index') is not None for i in infos),predicted_close_at_s=next((i['predicted_close_at_s'] for i in infos if i.get('predicted_close_at_s') is not None),None),gt_phase_cap_frame=next((i['gt_phase_cap_frame'] for i in infos if i.get('gt_phase_cap_frame') is not None),None),gt_contact_frame=next((i['gt_contact_frame'] for i in infos if i.get('gt_contact_frame') is not None),None),gt_cursor_minmax=[min(i['gt_cursor'] for i in infos),max(i['gt_cursor'] for i in infos)] if row['arm']=='A4' and infos else None,physical_abort=row['physical_abort']))
    groups={}
    for subset,arm in sorted({(r['subset'],r['arm']) for r in episodes}):
        rows=[r for r in episodes if r['subset']==subset and r['arm']==arm];groups[subset+'/'+arm]=dict(n=len(rows),close_entered=sum(r['close_entered'] for r in rows),zero_valid_episodes=sum(not r['valid_controls'] for r in rows),phase_entry_counts={p:sum(p in r['first_phase_tick'] for r in rows) for p in ('approach','close','lift','hold')},physical_aborts=dict(collections.Counter(r['physical_abort'] for r in rows if r['physical_abort'])))
    return dict(groups=groups,episodes=episodes,scope='Observed consumer phase; measured S2c success remains separately rescored. A4 no-close or invalid future is an implementation/admission finding, not an oracle upper bound.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mechanism-run',type=Path);a=p.parse_args();result=audit(a.run,a.mechanism_run);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['groups'],indent=2))
if __name__=='__main__':main()
