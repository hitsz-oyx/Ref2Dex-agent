"""Inject isolated in-memory errors; never mutate a source file or run physics."""
import sys,types,runpy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path.insert(0,str(TASK/'src'))
name='wam_rl_policy_v3.reference_admission'
source=(TASK/'src/wam_rl_policy_v3/reference_admission.py').read_text()
changes={
 'skip_reference_limits':('if bad or ((goal[self.fingers]<low-1e-7)|(goal[self.fingers]>high+1e-7)).any():','if False:'),
 'omit_passive_coupling':('for dst,src,ratio in COUPLINGS:','for dst,src,ratio in ():'),
 'no_measured_momentum_braking':('risk=self.mask&','risk=np.zeros(18,bool)&'),
 'send_unprotected_command':('return result,dict(braking=','return action,dict(braking='),
 'never_latch_stall':("if stagnant:self.latched='UNREACHABLE_STALL'","if False:self.latched='UNREACHABLE_STALL'"),
 'mix_near_and_outside':('outside=mask&((q<lo)|(q>hi))','outside=mask&((q<lo+.10)|(q>hi-.10))'),
 'certify_wrong_palm_chain':("if not np.allclose(palm[:3,3],q[:3],atol=1e-4,rtol=0):","if False:"),
 'skip_intermediate_trajectory':('for t in times:','for t in times[[0,-1]]:'),
}
results=[]
for label,(old,new) in changes.items():
 assert old in source,label
 module=types.ModuleType(name);module.__package__='wam_rl_policy_v3'
 exec(compile(source.replace(old,new),name,'exec'),module.__dict__);sys.modules[name]=module
 tests=runpy.run_path(str(TASK/'tests/test_reference_admission.py'));fail=[]
 # Object tracking imports the envelope inside __init__, so exercises the mutant.
 for key,function in tests.items():
  if key.startswith('test_'):
   try:function()
   except BaseException as error:fail.append(dict(test=key,error=type(error).__name__))
 results.append(dict(mutant=label,killed=bool(fail),detected_by=fail))
print(json.dumps(results,indent=2));assert all(x['killed'] for x in results),results
print('PASS killed %d/%d'%(len(results),len(results)))
