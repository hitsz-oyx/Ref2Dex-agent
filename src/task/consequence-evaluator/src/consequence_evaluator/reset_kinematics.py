"""URDF link states at reset without advancing the simulation clock."""
from pathlib import Path
import xml.etree.ElementTree as ET


class ResetKinematics:
    def __init__(self, urdf, dof_names, body_names, device):
        import torch
        from isaacgym import torch_utils
        self.torch, self.quat = torch, torch_utils
        xml = ET.parse(Path(urdf)).getroot()
        self.body_names = tuple(body_names)
        self.dof_names = tuple(dof_names)
        self.children = {}
        links = {link.get('name') for link in xml.findall('link')}
        child_links, movable = set(), set()
        for joint in xml.findall('joint'):
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            kind, name = joint.get('type'), joint.get('name')
            origin, axis = joint.find('origin'), joint.find('axis')
            def vector(element, attribute, default):
                value = default if element is None else element.get(attribute, default)
                return torch.tensor([float(v) for v in value.split()], device=device)
            position = vector(origin, 'xyz', '0 0 0')
            rpy = vector(origin, 'rpy', '0 0 0')
            rotation = torch_utils.quat_from_euler_xyz(rpy[0], rpy[1], rpy[2])
            direction = vector(axis, 'xyz', '1 0 0')
            if kind != 'fixed':
                if kind not in ('prismatic', 'revolute', 'continuous') or name not in self.dof_names:
                    raise ValueError('unsupported reset joint: '+name)
                movable.add(name)
                direction = direction / direction.norm()
            index = -1 if kind == 'fixed' else self.dof_names.index(name)
            self.children.setdefault(parent, []).append((child, kind, position, rotation, direction, index))
            child_links.add(child)
        roots = links-child_links
        if len(roots) != 1 or movable != set(self.dof_names) or not set(body_names) <= links:
            raise ValueError('native actor/URDF reset topology mismatch')
        self.root = roots.pop()

    def states(self, q, qdot, root):
        torch, quat = self.torch, self.quat
        if q.shape != qdot.shape or q.shape[-1] != len(self.dof_names):
            raise ValueError('reset q/qdot shape mismatch')
        batch = len(q)
        states = {self.root: (root[:,:3], root[:,3:7], root[:,7:10], root[:,10:13])}
        def visit(parent):
            pp, pr, pv, pw = states[parent]
            for child, kind, position, rotation, direction, index in self.children.get(parent, ()):
                jr = quat.quat_mul(pr, rotation.expand(batch, 4))
                axis = direction.expand(batch, 3)
                world_axis = quat.quat_rotate(jr, axis)
                delta = quat.quat_rotate(pr, position.expand(batch, 3))
                cr = jr
                cv = pv + torch.cross(pw, delta, dim=-1)
                cw = pw
                if kind == 'prismatic':
                    motion = world_axis*q[:,index,None]
                    delta = delta+motion
                    cv = cv+torch.cross(pw, motion, dim=-1)+world_axis*qdot[:,index,None]
                elif kind in ('revolute', 'continuous'):
                    cr = quat.quat_mul(jr, quat.quat_from_angle_axis(q[:,index], axis))
                    cw = pw+world_axis*qdot[:,index,None]
                states[child] = (pp+delta, cr, cv, cw)
                visit(child)
        visit(self.root)
        return torch.stack([torch.cat(states[name], dim=-1) for name in self.body_names], dim=1)


def task_kinematics(task):
    if not hasattr(task, '_consequence_reset_fk'):
        names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
        dofs = task.gym.get_actor_dof_names(task.envs[0], task.humanoid_handles[0])
        urdf = Path(task.cfg['env']['asset']['assetRoot'])/task.robot_type
        task._consequence_reset_fk = ResetKinematics(urdf, dofs, names, task._dof_pos.device)
    return task._consequence_reset_fk
