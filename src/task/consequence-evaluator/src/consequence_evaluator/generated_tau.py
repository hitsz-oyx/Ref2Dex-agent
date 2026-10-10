"""Frozen measured-history proposals, score selection and native projection."""
import numpy as np
import torch

from .proposal_history import condition
from .proposal_runtime import TauProposal, retrieve_rows, choose_generated
from .trajectory_utility import TrajectoryUtility
from .tau_projection import project_tau
from .reference_tracking import reference_velocity, future_reference_velocity


class GeneratedTau:
    def __init__(self, proposal_packet, evaluator_packet, train_bank, urdf, device, dt,
                 projection_iterations=300,positions_only=False,cuda_graph=False):
        if projection_iterations not in (60,300): raise ValueError('declared bounded projection required')
        self.projection_iterations=projection_iterations
        self.positions_only=positions_only
        if cuda_graph and not positions_only:raise ValueError('capture requires position FK')
        self.graph_cache={} if cuda_graph else None
        self.device = device; self.urdf = urdf; self.dt = dt
        self.proposal = TauProposal(proposal_packet).to(device).eval()
        if evaluator_packet.get('arm') != 'T':
            raise ValueError('frozen tau-only evaluator required')
        a = evaluator_packet['architecture']
        self.evaluator = TrajectoryUtility(a['history_dim'], a['width'], a['layers']).to(device).eval()
        self.evaluator.load_state_dict(evaluator_packet['model'])
        self.mean, self.scale = [torch.as_tensor(v, device=device) for v in evaluator_packet['statistics']['trajectory']]
        with torch.no_grad():
            self.bank = self.proposal.encode(torch.as_tensor(train_bank['history'], device=device))
        self.episode = torch.as_tensor(np.unique(train_bank['episode'], return_inverse=True)[1], device=device)
        self.displacement = torch.as_tensor(train_bank['target']-train_bank['current'][:, None], device=device)

    def propose(self, history, roles):
        """Exactly four measured states, no clock/phase/reference/force argument."""
        features=[]; current=[]
        for i in range(len(roles)):
            h, hand=condition(history['obj'][:,i],history['hand'][:,i],history['q'][:,i],
                history['dq'][:,i],history['velocity'][:,i])
            features.append(h); current.append(hand)
        h=torch.as_tensor(np.stack(features),device=self.device)
        current=torch.as_tensor(np.stack(current),device=self.device)
        with torch.no_grad():
            learned=self.proposal(h,current)
            ids, distances=retrieve_rows(self.proposal.encode(h),self.bank,self.episode,8)
            retrieval=self.displacement[ids]+current[:,None,None]
            pool=torch.cat((current[:,None,None].expand(-1,1,24,-1,-1),learned[:,None],retrieval),1)
            trajectories=(pool.reshape(-1,24,33)-self.mean)/self.scale
            scores=self.evaluator(torch.zeros(len(trajectories),1442,device=self.device),trajectories,
                torch.zeros(len(trajectories),24,12,device=self.device),False).reshape(-1,10).cpu().numpy()
            choices=choose_generated(scores)
            for i, role in enumerate(roles):
                if role == 'persistence': choices[i]=0
                elif role == 'displacement': choices[i]=1
                elif role != 'scored': raise ValueError('unknown generated role')
            raw=pool[torch.arange(len(roles),device=self.device),torch.as_tensor(choices,device=self.device)].cpu().numpy()
        return dict(history=np.stack(features),current=current.cpu().numpy(),raw=raw,
            choices=choices,score=scores,source_rows=ids.cpu().numpy(),distances=distances.cpu().numpy())

    def project_selected(self, history, audit):
        """Project already selected tau; privileged diagnostics stay outside proposal."""
        pose=history['obj'][-1]
        world=np.einsum('nij,ntpj->ntpi',pose[:,:3,:3],audit['raw'])+pose[:,None,None,:3,3]
        # Geometry is fitted without dynamics/labels. Repaired native points,
        # rather than the nonrigid prediction, condition the frozen controller.
        with torch.enable_grad():
            fit=project_tau(history['hand'][-1],world,history['q'][-1],self.urdf,
                            self.device,iterations=self.projection_iterations,deadline_s=30,
                            positions_only=self.positions_only,graph_cache=self.graph_cache)
        q=torch.as_tensor(fit['q'],device=self.device)
        # q[0] is measured calibration, not a nominal future target. Default
        # FF excludes it; only explicit diagnostic controls retain old velocity.
        future_velocity=np.asarray(audit.get('future_velocity',np.ones(len(q),dtype=bool)))
        velocity=torch.stack([(future_reference_velocity(v,self.dt) if future_velocity[i]
                              else reference_velocity(v,self.dt)) for i,v in enumerate(q)])
        audit=dict(audit,future_velocity=future_velocity)
        hand=torch.as_tensor(fit['points'][:,1:],device=self.device)
        fitted_hand=hand.cpu().numpy().copy()
        if 'condition_raw_tau' in audit:
            mask=torch.as_tensor(audit['condition_raw_tau'],device=self.device)
            hand[mask]=torch.as_tensor(world,device=self.device)[mask]
        audit=dict(audit,hand=hand.cpu().numpy(),fitted_hand=fitted_hand,q=fit['q'],projection_s=fit['elapsed_s'])
        return dict(hand=hand,q=q,velocity=velocity,audit=audit)

    def plan(self, history, roles):
        return self.project_selected(history,self.propose(history,roles))
