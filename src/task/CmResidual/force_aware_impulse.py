"""Conservation-based object impulse; current force is an approximate prior."""
import numpy as np
import torch

SCHEMA='current70-weightforce18-prior3-action12-nongravity-impulse-v1'

def impulse_target(current_velocity,next_velocity,gravity,dt):
    scale=float(np.linalg.norm(gravity));assert scale>0 and dt>0
    return (next_velocity-current_velocity)/(scale*dt)-np.asarray(gravity)/scale

def force_prior(current_force,current_velocity,mass,gravity,damping=.01):
    scale=float(np.linalg.norm(gravity));return current_force/(mass[...,None]*scale)-damping*current_velocity/scale

def initialized_model(seed=2191):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);m=torch.nn.Sequential(torch.nn.Linear(103,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,3))
        torch.nn.init.zeros_(m[-1].weight);torch.nn.init.zeros_(m[-1].bias)
    return m
