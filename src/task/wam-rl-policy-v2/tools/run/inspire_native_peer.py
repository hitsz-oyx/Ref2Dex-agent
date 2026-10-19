"""Reuse the audited native collector with a bounded WAMv4 policy hook.

Only Task-local runtime adaptation: native initialization/capture/PD audit remain
in collect_rl660_batch. Official weights load only for the positive control.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT/'src/task/ptv3-shared-wam/tools/run/collect_rl660_batch.py'


def shared_preflight(gpu, server_pid):
    rows = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    for row in rows.splitlines():
        pid, memory = [int(v.strip()) for v in row.split(',')]
        if pid == server_pid:
            continue
        owner = subprocess.check_output(['ps','-o','user=','-p',str(pid)],text=True).strip()
        if owner != 'monica' or memory > 1024:
            raise RuntimeError('unexpected GPU occupant: '+row+' '+owner)


def adapted_source():
    source = SOURCE.read_text()
    def replace(old,new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('collector adaptation no longer applies: '+old[:100])
        source = source.replace(old,new)
    replace("    args = parser.parse_args()", "    parser.add_argument('--socket', type=Path)\n    parser.add_argument('--replay', type=Path)\n    parser.add_argument('--server-pid', type=int, default=-1)\n    parser.add_argument('--controls', type=int, default=300)\n    args = parser.parse_args()\n    if not 1 <= args.controls <= 600: raise ValueError('bounded controls required')")
    replace("    allowed = [ROOT/'outputs/ptv3-shared-wam']", "    allowed = [ROOT/'outputs/wam-rl-policy-v2']")
    replace("    if occupied:\n        raise RuntimeError('assigned GPU occupied: '+occupied)", "    shared_preflight(args.gpu,args.server_pid)")
    replace("    from isaacgym import gymtorch", "    import socket\n    from isaacgym import gymtorch")
    replace("    from trajectory_policy.wam_shadow import NATIVE_LINKS", "    from trajectory_policy.wam_shadow import NATIVE_LINKS\n    from trajectory_policy.wam_ipc import send, receive, peer_pid\n    from consequence_evaluator.physical_geometry import PhysicalGeometry\n    from consequence_evaluator.value_geometry import TableSupport")
    replace("        actor_role='official data generator only; no official weights in learned WAM/Chunk-R')", "        actor_role=('self-trained WAMv4-conditioned BC; no teacher weights or takeover' if args.socket else 'recorded canonical action replay positive control' if args.replay else 'official teacher positive control only'),\n        collector_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),native_adapter='inspire_native_peer.v1')")
    replace("        def restore(self,filename):", "        def restore(self,filename):\n            if args.socket or args.replay:\n                return  # learned/replay roles never load official actor weights")
    replace("        @torch.no_grad()\n        def _run_export_rl(self,task):", """        def policy_packet(self, final=False):
            task = self.inspire_task
            n = task.num_envs
            if n != 1: raise ValueError('single motion engineering probe only')
            names = task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0])
            ids = [names.index(name) for name in NATIVE_LINKS]
            packet = dict(schema='ref2dex.inspire-v4-ipc.v1', final=final,
                tick=int(task.progress_buf[0]),
                current=task._compute_observations_iter(torch.arange(n,device=task.device),0)[0].cpu().tolist(),
                pose=poses(task._target_states)[0].cpu().tolist(),
                hand=task._rigid_body_state.view(n,-1,13)[0,ids,:3].cpu().tolist())
            send(self.inspire_connection,packet)
            reply = receive(self.inspire_connection)
            if reply.get('tick') != packet['tick']: raise ValueError('stale IPC action')
            return reply

        def get_action(self,obs,is_deterministic=False):
            if args.replay:
                if not hasattr(self,'replay_actions'):
                    self.replay_actions=np.load(args.replay)['action'].copy()
                tick=int(self.inspire_task.progress_buf[0])
                return torch.tensor(self.replay_actions[tick:tick+1],device=self.device)
            if not args.socket: return super().get_action(obs,is_deterministic)
            reply = self.policy_packet()
            return torch.tensor(reply['action'],device=self.device,dtype=torch.float32).reshape(1,18)

        @torch.no_grad()
        def _run_export_rl(self,task):
            self.inspire_task = task
            if args.socket:
                self.inspire_connection = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
                self.inspire_connection.settimeout(90)
                self.inspire_connection.connect(str(args.socket))
                if peer_pid(self.inspire_connection) != args.server_pid: raise ValueError('wrong WAM server')""")
    replace("            assets = asset_view", "            assets = asset_view\n            physical_geometry = PhysicalGeometry(task,assets,distance_device=task.device)\n            table_support = TableSupport(assets,task.device)")
    replace("            capture = actual_pd = None", "            logs.update(surface_gap=[],support_gap=[],table_footprint=[])\n            state_fields.update(('surface_gap','support_gap','table_footprint'))\n            capture = actual_pd = None")
    replace("                for key,value in values.items():", "                _,gap = physical_geometry.measure(task)\n                support,footprint = table_support.measure(task,physical_geometry)\n                values.update(surface_gap=gap,support_gap=support,table_footprint=footprint)\n                for key,value in values.items():")
    replace("            current = record(0)", "            maximum = min(maximum,args.controls+1)\n            lengths = lengths.clamp(max=maximum)\n            current = record(0)")
    replace("            data = {k:np.stack(v) for k,v in logs.items()}", "            if args.socket:\n                self.policy_packet(final=True)\n                self.inspire_connection.close()\n            data = {k:np.stack(v) for k,v in logs.items()}")
    return source


if __name__ == '__main__':
    # Reuse the original module root resolution, not this wrapper's Task path.
    context = dict(__name__='__main__',__file__=str(SOURCE), shared_preflight=shared_preflight, SOURCE=SOURCE,hashlib=hashlib)
    exec(compile(adapted_source(),str(SOURCE),'exec'),context)
