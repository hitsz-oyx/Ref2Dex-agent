"""Freeze n40 truly interface-unseen initial families, without policy-outcome selection."""
import argparse,collections,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path.insert(0,str(TASK/'tools/run'))
from prepare_large_grasp_panel import load_raw
sys.path.insert(0,str(TASK/'tools/audit'))
from freeze_grasp_geometry import copy_geometry_runtime


def codes(value):
    if isinstance(value,str):
        if not value.startswith('dexplore/'):return set()
        name,separator,scale=value.partition('@')
        if not separator:return {name}
        # Only the requested native unit scale is eligible; never alias another
        # scale under the requested cache key.
        try:return {name} if float(scale)==1. else set()
        except ValueError:return set()
    if isinstance(value,dict):return set().union(*(codes(k)|codes(v) for k,v in value.items())) if value else set()
    if isinstance(value,list):return set().union(*(codes(v) for v in value)) if value else set()
    return set()


def select(catalog,cached,exposed):
    groups=collections.defaultdict(list)
    for r in sorted(catalog,key=lambda r:r['sequence']):
        if r['engineering']=='PASS' and 'dexplore/'+r['object'] in cached and r['object'] not in exposed:groups[r['object']].append(r)
    objects=sorted(o for o,rows in groups.items() if len({r['sequence'] for r in rows})>=2)[:20]
    if len(objects)!=20:raise ValueError('need 20 unexposed cached objects with two PASS source sequences; got '+str(len(objects)))
    return [groups[o][j] for j in (0,1) for o in objects]


def assignments(cases):
    arms=['A1','A2','A0'];jobs=[]
    if len(cases)!=40 or len({r['sequence'] for r in cases})!=40 or len({r['object'] for r in cases})!=20:raise ValueError('40 distinct initials /20 objects required')
    for i,r in enumerate(cases):
        rotation=(i//3+i%3)%3
        jobs.append(dict(id='p%03d'%i,subset='interface_unseen',sequence=r['sequence'],object=r['object'],reference=r['reference'],original_split=r['split'],initial_family=r['sequence'],replicate=0,native_seed=100+i,model_seed=180+i,worker=i%3,order=arms[rotation:]+arms[:rotation]))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents:raise ValueError('fresh task output required')
    if shutil.disk_usage(ROOT).free<20*2**30:raise ValueError('storage safety reserve')
    old=json.loads((ROOT/'outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/registration.json').read_text());small=json.loads((ROOT/'outputs/wam-rl-policy-v3/grasp-v11-coverage-20261022-r1/registration.json').read_text())
    exposed={'cylindersmall','airplane','apple','cubelarge','duck','mug'}|{j['object'] for j in old['jobs']}|{j['object'] for j in small['jobs']}
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'];cached=codes(json.loads((ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry/surface-manifest.json').read_text()))
    selected=select(catalog,cached,exposed);out.mkdir(parents=True);cases=[]
    import numpy as np
    for r in selected:
        ref,points,table,bounds=load_raw(r);folder=out/'references'/r['sequence'];folder.mkdir(parents=True);np.savez_compressed(folder/'reference.npz',**ref)
        cases.append(dict(r,reference=str(folder/'reference.npz')))
    frozen=out/'frozen-source';copy_geometry_runtime(frozen)
    # Keep the actual evaluator and its legacy freeze helper reproducible while
    # leaving their upstream scientific algorithms unchanged.
    audit=frozen/'src/task/wam-rl-policy-v3/tools/audit';audit.mkdir(parents=True,exist_ok=True)
    for name in ('repeat_frozen_tau_chains.py','measure_s2d_controls.py','freeze_grasp_geometry.py'):shutil.copyfile(TASK/'tools/audit'/name,audit/name)
    (frozen/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/tau_replay_contract.py').write_bytes((TASK/'src/wam_rl_policy_v3/tau_replay_contract.py').read_bytes())
    helper=audit/'freeze_grasp_geometry.py';source=helper.read_text().replace("ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'","ROOT=Path("+repr(str(frozen))+");TASK=ROOT/'src/task/wam-rl-policy-v3'");helper.write_text(source)
    legacy=audit/'repeat_frozen_tau_chains.py';source=legacy.read_text().replace("ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'","ROOT=Path("+repr(str(ROOT))+");TASK=Path("+repr(str(frozen/'src/task/wam-rl-policy-v3'))+")");legacy.write_text(source)
    driver=(TASK/'tools/run/run_grasp_geometry_worker.py').read_text();needle="ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'"
    if driver.count(needle)!=1:raise ValueError('worker root seam changed')
    driver=driver.replace(needle,"ROOT=Path("+repr(str(ROOT))+");TASK=Path("+repr(str(frozen/'src/task/wam-rl-policy-v3'))+")");(out/'frozen-worker.py').write_text(driver)
    reg=dict(protocol='grasp.geometry-SE3.interface-unseen40.3arms.v1',experiment_id='P-20261022-grasp-coverage40',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),jobs=assignments(cases),arms=['A1','A2','A0'],subsets=['interface_unseen'],n_per_arm_per_subset=40,horizon_controls=600,control_hz=30,worker_wall_s=14400,maximum_controls=72000,learning_updates=0,ppo_authorized=False,release_applicable=False,
        primary='full600, measured geometry S2c current.hold, object goal error<=0.10*minimum convex-hull width, no physical abort',main_rule='A1>A2 iff two-sided Fisher p<.05 AND Wilson95(A1).lower>Wilson95(A2).upper; old v10 seen reported separately',
        source_control='A0 source-PD approach then shared SE3 servo; separate physical witness, NOT GT-tau/R upper bound; source admission/fallback remains in denominator',
        excluded_exposed_objects=sorted(exposed),selected_objects=sorted({r['object'] for r in selected}),selection='first20 lexical previously unexposed cached objects with >=2 engineering-PASS sequences; first2 sequences each; no policy outcome/hold/precision filter',
        scope='40 distinct interface-unseen initials,20 new objects x2 sequences. Not pretraining-unseen; no pooling with seen or 12-case engineering run. Conditional fixed-panel Wilson/Fisher; paired supplement separately.',
        future_goal_contract=dict(frame='current_object',training='actual executed t+24 endpoint',online='scheduler desired waypoint; no executed-t+24 guarantee',valid='query mask, all32 True; not a GT-future flag'),
        runtime_source=str(frozen),worker_source=str(out/'frozen-worker.py'),missing='technical errors preserve partial budget and same assigned initial/seed; physical failures never deleted')
    (out/'registration.json').write_text(json.dumps(reg,indent=2)+'\n');print('REGISTERED 40 distinct initials /20 unexposed objects x3 arms=120 episodes; <=72000 controls; 0 updates')


if __name__=='__main__':main()
