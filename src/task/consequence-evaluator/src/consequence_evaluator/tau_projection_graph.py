"""Capture repeated geometric FK/backward; leave ordinary Adam outside capture."""
import time
import torch

from .contracts import HAND_LINKS
from .reference_motion import NATIVE_DOF_NAMES
from .reference_tracking import apply_coupling
from .reset_kinematics import ResetKinematics
from .tau_tracking import FINGERS,FINGER_LIMITS


class FingerFitGraph:
    """One fixed-shape graph, reused with fresh live-query buffers each plan.

    Only FK, per-frame best updates and backward are captured. Adam retains
    its original CPU step/bias corrections and is reset for every independent
    fit. This avoids changing optimization arithmetic to capturable Adam.
    """
    def __init__(self,fixed,target,initial,urdf):
        if not fixed.is_cuda or torch.__version__!='2.4.1+cu121':
            raise ValueError('pinned CUDA runtime required for geometry capture')
        self.fixed=fixed.clone();self.target=target.clone()
        self.variable=initial.clone().requires_grad_(True)
        self.variable.grad=torch.zeros_like(self.variable)
        self.limits=torch.tensor(FINGER_LIMITS,device=fixed.device)
        self.finger_ids=torch.tensor(FINGERS,device=fixed.device)
        self.fk=ResetKinematics(urdf,NATIVE_DOF_NAMES,HAND_LINKS,fixed.device)
        self.root=torch.zeros(len(fixed),13,device=fixed.device);self.root[:,6]=1
        self.best=initial.clone()
        with torch.no_grad():self.best_error=(self.points()-self.target).square().sum((1,2))
        self.optimizer=torch.optim.Adam([self.variable],lr=.025)
        self.graph=torch.cuda.CUDAGraph()
        stream=torch.cuda.Stream(device=fixed.device)
        stream.wait_stream(torch.cuda.current_stream(fixed.device))
        with torch.cuda.stream(stream):
            for _ in range(3):self.gradient()
        torch.cuda.current_stream(fixed.device).wait_stream(stream)
        torch.cuda.synchronize(fixed.device)
        with torch.cuda.graph(self.graph):self.gradient()

    def coupled(self):
        q=self.fixed.clone();q[:,self.finger_ids]=self.variable
        return apply_coupling(q)

    def points(self):
        return self.fk.positions(self.coupled(),self.root)

    def gradient(self):
        self.variable.grad.zero_()
        error=(self.points()-self.target).square().sum((1,2))
        with torch.no_grad():
            improved=error<self.best_error
            # Fixed shapes avoid boolean indexing's CUDA->CPU nonzero sync.
            self.best.copy_(torch.where(improved[:,None],self.variable,self.best))
            self.best_error.copy_(torch.minimum(self.best_error,error))
        error.mean().backward()

    def fit(self,fixed,target,initial,iterations,started,deadline_s,monitor):
        if fixed.shape!=self.fixed.shape or target.shape!=self.target.shape or initial.shape!=self.variable.shape:
            raise ValueError('geometry capture cache shape changed')
        with torch.no_grad():
            self.fixed.copy_(fixed);self.target.copy_(target);self.variable.copy_(initial)
            self.variable.clamp_(min=0);self.variable.copy_(torch.minimum(self.variable,self.limits))
            self.best.copy_(self.variable)
            self.best_error.copy_((self.points()-self.target).square().sum((1,2)))
        self.optimizer.state.clear()  # Each query starts with zero moments/step.
        for i in range(iterations):
            if time.monotonic()-started>deadline_s:raise TimeoutError('captured projection deadline')
            self.graph.replay()
            self.optimizer.step()
            with torch.no_grad():
                self.variable.clamp_(min=0);self.variable.copy_(torch.minimum(self.variable,self.limits))
            if monitor and (i==0 or (i+1)%50==0):monitor(i+1,time.monotonic()-started)
        with torch.no_grad():
            error=(self.points()-self.target).square().sum((1,2))
            self.best.copy_(torch.where((error<self.best_error)[:,None],self.variable,self.best))
            self.best_error.copy_(torch.minimum(self.best_error,error))
            q=self.fixed.clone();q[:,self.finger_ids]=self.best
            return apply_coupling(q),self.best_error
