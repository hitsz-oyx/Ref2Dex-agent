"""Causal, read-only stage evidence and bounded reward for v6 execution.

Inputs are existing RewardFeatures, measured interaction feedback and native
contact forces. No setters, reference trajectory or predicted WAM output enter.
"""
import numpy as np

STAGE_CONTRACT = 'v6.approach-contact-lift5cm-hold1s-release.v1'
REWARD_CONTRACT = 'v6.staged-highwater-dense.existing-parts.v1'
STAGES = ('approach', 'contact', 'lift', 'hold', 'release')
WEIGHTS = np.array([.1, .15, .25, .25, .25])


class StagedProgress:
    def __init__(self, initial_z, *, max_sample_gap_s=1/30+1e-8):
        if not np.isfinite([initial_z, max_sample_gap_s]).all() or max_sample_gap_s <= 0:
            raise ValueError('finite initial height and positive sample gap required')
        self.initial_z = float(initial_z)
        self.max_sample_gap_s = max_sample_gap_s
        self.first = dict.fromkeys(STAGES)
        self.last_time = self.contact_z = self.hold_since = self.release_since = None
        self.longest_hold_s = 0.
        self.progress = np.zeros(5)
        self.current = dict.fromkeys(STAGES, False)

    def update(self, z, now, features, feedback, contact_forces, *,
               release_authorized=False, physical_abort=False):
        if not np.isfinite([z, now]).all() or (self.last_time is not None and now <= self.last_time):
            raise ValueError('strictly increasing finite stage time required')
        forces = np.asarray(contact_forces, dtype=float)
        if forces.shape != (5, 3):
            raise ValueError('five contact force vectors required')
        gaps = np.array([features.palm_gap_m, features.mean_tip_gap_m, features.hand_clearance_m])
        if not np.isfinite(gaps).all() or (gaps < 0).any():
            raise ValueError('finite nonnegative existing distance features required')
        if self.last_time is not None and now-self.last_time > self.max_sample_gap_s:
            self.hold_since = self.release_since = None
        self.last_time = float(now)
        safe = bool(not physical_abort and feedback.workspace_ok)
        valid = bool(feedback.force_valid and np.isfinite(forces).all())
        counts = int((np.linalg.norm(forces, axis=-1) >= .1).sum()) if valid else None
        approach = bool(safe and gaps[0] <= .10 and gaps[1] <= .05)
        contact = bool(safe and approach and valid and counts >= 2 and feedback.contact_observed)
        if approach and self.first['approach'] is None:
            self.first['approach'] = float(now)
        if contact and self.first['contact'] is None:
            self.first['contact'] = float(now)
            self.contact_z = max(self.initial_z, float(z))
        after_contact = self.first['contact'] is not None and now > self.first['contact']
        coupled = bool(contact and feedback.grasp_established and not feedback.grasp_lost
                       and feedback.supported is False and after_contact)
        lift_fraction = float(np.clip((z-self.contact_z)/.05, 0, 1)) if coupled else 0.
        lifted = bool(coupled and z-self.contact_z >= .05-1e-10)
        if lifted and self.first['lift'] is None:
            self.first['lift'] = float(now)
        if lifted:
            if self.hold_since is None:
                self.hold_since = float(now)
            hold_s = float(now)-self.hold_since
            self.longest_hold_s = max(self.longest_hold_s, hold_s)
        else:
            self.hold_since = None
            hold_s = 0.
        held = bool(lifted and hold_s >= 1.-1e-10)
        if held and self.first['hold'] is None:
            self.first['hold'] = float(now)
        # Feedback.released uses MIN tip gap >=25mm; mean gap alone is unsafe.
        # Force validity is mandatory: missing measurements never mean no force.
        detached = bool(safe and release_authorized and self.first['hold'] is not None
                        and now > self.first['hold'] and valid and counts == 0
                        and feedback.released and not feedback.grasp_established
                        and features.hand_clearance_m >= .025)
        if detached:
            if self.release_since is None:
                self.release_since = float(now)
        else:
            self.release_since = None
        released = bool(detached and now-self.release_since >= .12-1e-10)
        if released and self.first['release'] is None:
            self.first['release'] = float(now)
        self.current = dict(zip(STAGES, (approach, contact, lifted, held, released)))
        proximity = float(np.clip(1-(gaps[0]/.10+gaps[1]/.05)/2, 0, 1)) if safe else 0.
        self.progress = np.array([proximity, float(contact), lift_fraction,
                                  min(hold_s, 1.), float(released)])
        return self.snapshot()

    def snapshot(self):
        return dict(contract=STAGE_CONTRACT, current=self.current.copy(),
                    achieved={k: self.first[k] is not None for k in STAGES},
                    first_time_s=self.first.copy(), longest_hold_s=self.longest_hold_s,
                    continuous_hold_s=0. if self.hold_since is None else self.last_time-self.hold_since)


class StagedDenseReward:
    """High-water increments cannot pay repeatedly for cyclic partial progress."""
    def __init__(self, stages):
        self.highwater = stages.progress.copy()  # initial state earns no free reward
        self.paid = {k for k in STAGES if stages.first[k] is not None}
        self.terminal = False
        self.totals = dict.fromkeys(('shaping', 'event', 'sparse', 'time', 'effort'), 0.)

    def step(self, stages, decision, action, *, terminal=False, failed=False, dt_s=1/30):
        action = np.asarray(action, dtype=float)
        if self.terminal or action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('live episode and finite canonical action12 required')
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError('positive finite elapsed control duration required')
        updated = np.maximum(self.highwater, stages.progress)
        increments = WEIGHTS*(updated-self.highwater)
        self.highwater = updated
        events = [k for k in STAGES if stages.first[k] is not None and k not in self.paid]
        self.paid.update(events)
        failed = bool(failed or decision.failure_reason)
        # Physical abort/external reset overrides a simultaneous scheduler success.
        sparse = -1. if failed else 1. if decision.task_complete else 0.
        parts = dict(shaping=float(increments.sum()), event=.1*len(events), sparse=sparse,
                     time=-.01*dt_s, effort=-.01*float(np.mean(np.square(action.clip(-1, 1)))))
        self.terminal = bool(terminal or decision.terminated or decision.task_complete or failed)
        for key, value in parts.items():
            self.totals[key] += value
        return float(sum(parts.values())), parts, dict(zip(STAGES, increments.tolist()))
