"""Real WAM/full-BC boundary on archived TRAIN measurements; no physics execution."""
import argparse,json,os,signal,sys,time
from pathlib import Path
import numpy as np
import torch
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.bc_mlp import CurrentMLP,SCHEDULED_SCHEMA
from wam_rl_policy_v3.control_contracts import MeasuredShadowState,TaskIntent,scheduled_conditioning_contract
from wam_rl_policy_v3.control_pipeline import ScheduledShadowControl
from wam_rl_policy_v3.goal_scheduler import OrderedGoalScheduler
from wam_rl_policy_v3.scheduled_executor import BCWarmStartActor
from wam_rl_policy_v3.wam_v4_runtime import V4Runtime


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',default='6');a=p.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;torch.set_num_threads(2)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120s engineering limit')));signal.alarm(120)
 a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 cache=ROOT/'outputs/shadow-finetune/policy-data-full-20261015-r1';snap=ROOT/'outputs/ptv3-shared-wam-v4/local-control-snapshot-20261018-r1'
 row=json.loads((cache/'manifest.json').read_text())['records'][0]
 assert row['split']=='train'
 data=dict(np.load(cache/row['path']))
 runtime=V4Runtime(snap/'step12000-local.pt',[row],cache,dense_manifest=snap/'surface-manifest.json')
 parent=torch.load(ROOT/'outputs/wam-rl-policy/shadow-current-bc-seed241-20261016-r1/best.pt',map_location='cpu',weights_only=False)
 old=CurrentMLP().cuda().eval();old.load_state_dict(parent['model'])
 expanded=CurrentMLP(565);expanded.x_scale[300:564]=.05;expanded.initialize_current(parent)
 actor=BCWarmStartActor(dict(schema=SCHEDULED_SCHEMA,model=expanded.state_dict(),config=dict(conditioning=scheduled_conditioning_contract()))).cuda().eval()
 target=data['poses'][0].copy();target[:3,3]=data['goal'][0]
 scheduler=OrderedGoalScheduler([data['poses'][0],target],[0,2],runtime.geometry[0][2],interaction_points=runtime.geometry[0][0])
 control=ScheduledShadowControl(runtime,scheduler,actor,TaskIntent('archived-lift'));control.reset('archived-0')
 records=[];max_parent_error=max_logp_error=0.;startup=[]
 for tick in range(12):
  state=MeasuredShadowState('archived-0',tick,float(data['time'][tick]),data['observation'][tick],data['poses'][tick],data['hand'][tick],np.ones(11,bool))
  command=control.compute(state,sample=True)
  x=torch.as_tensor(command['executor_observation'][None],device='cuda')
  raw=torch.as_tensor(command['raw_action'][None],device='cuda')
  with torch.no_grad():
   max_parent_error=max(max_parent_error,float((actor(x)-old(torch.as_tensor(data['observation'][tick:tick+1],device='cuda'))).abs().max()))
   max_logp_error=max(max_logp_error,abs(float(actor.distribution(x).log_prob(raw).sum())-command['action_log_prob']))
  before=command['canonical_action'].copy()
  def mutating_sink(action):action[:,:6].mul_(3)
  control.execute(command,mutating_sink)
  np.testing.assert_array_equal(before,command['canonical_action'])
  startup.append(bool(command['tau_contract']['valid'].all()))
  records.append(command)
 assert startup==[False]*6+[True]*6 and max_parent_error<1e-5 and max_logp_error<1e-5
 np.savez_compressed(a.output/'trace.npz',observation=np.stack([r['executor_observation'] for r in records]),tau=np.stack([r['tau'] for r in records]),canonical_action=np.stack([r['canonical_action'] for r in records]),raw_action=np.stack([r['raw_action'] for r in records]))
 report=dict(status='PASS',scope='12 archived TRAIN measurements; real WAM and full BC; mutating test sink, no native physics',
  source_record=row['path'],wam_step=runtime.step,parent_bc_step=parent['step'],max_parent_mean_error=max_parent_error,max_logprob_replay_error=max_logp_error,
  startup_valid=startup,goal_semantics=records[-1]['goal']['semantics'],goal_reach_time_guaranteed=False,phase_conditioning=False,
  elapsed_s=time.monotonic()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,python=sys.executable,torch=torch.__version__,gpu=a.gpu)
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));signal.alarm(0)
if __name__=='__main__':main()
