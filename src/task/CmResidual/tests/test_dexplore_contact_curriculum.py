"""Near-contact training reset is reproducible and preserves per-motion state."""
import torch

from src.task.CmResidual.dexplore_contact_curriculum import install_contact_reset_curriculum


def _task_class(contact_frames, num_envs, num_frames=24):
    class FakeTask:
        def __init__(self):
            self.device = "cpu"
            self.num_dof = 18
            self.hoi_data_dict = []
            for frame in contact_frames:
                contact = torch.zeros(num_frames, 1)
                if frame is not None:
                    contact[frame:] = 1
                self.hoi_data_dict.append({"contact": contact})
            self.max_episode_length = torch.tensor([num_frames] * len(contact_frames))
            self.hoi_refs = torch.zeros(len(contact_frames), 1, num_frames, 155)
            for motion in range(len(contact_frames)):
                for frame in range(num_frames):
                    self.hoi_refs[motion, 0, frame, 119:137] = motion * 100 + frame
                    self.hoi_refs[motion, 0, frame, 137:155] = frame * 2
            self.data_id = torch.zeros(num_envs, dtype=torch.long)
            self.ref_index = torch.zeros(num_envs, dtype=torch.long)
            self.progress_buf = torch.zeros(num_envs, dtype=torch.long)
            self.start_times = torch.zeros(num_envs, dtype=torch.long)
            self._hist_obs = torch.ones(num_envs, 2)
            self.contact_reset = torch.ones(num_envs, 3)
            self._dof_pos = torch.zeros(num_envs, 18)
            self._dof_vel = torch.zeros(num_envs, 18)

        def _reset_ref_state_init(self, env_ids):
            self.data_id[env_ids] = env_ids % len(contact_frames)

        def _set_env_state(self, env_ids, dof_pos, dof_vel):
            self._dof_pos[env_ids] = dof_pos
            self._dof_vel[env_ids] = dof_vel

    return FakeTask


def test_contact_resets_use_own_motion_and_only_reset_selected_envs():
    cls = _task_class([3, 9], 4)
    install_contact_reset_curriculum(cls, before=0, after=0)
    task = cls()
    task._reset_ref_state_init(torch.tensor([0, 1, 3]))
    assert task.progress_buf.tolist() == [3, 9, 0, 9]
    assert task.start_times.tolist() == [3, 9, 0, 9]
    assert task._dof_pos[:, 0].tolist() == [3.0, 109.0, 0.0, 109.0]
    assert task._dof_vel[:, 0].tolist() == [6.0, 18.0, 0.0, 18.0]
    assert task._hist_obs[2].eq(1).all() and task._hist_obs[[0, 1, 3]].eq(0).all()


def test_contact_window_samples_both_sides_and_clamps_episode_end():
    cls = _task_class([21], 256)
    install_contact_reset_curriculum(cls, before=4, after=4)
    task = cls()
    torch.manual_seed(42)
    task._reset_ref_state_init(torch.arange(256))
    assert task.progress_buf.min() >= 17
    assert task.progress_buf.max() <= 22  # final valid reference is length - 2
    assert task.progress_buf.unique().numel() >= 3


def test_contact_curriculum_refuses_motion_without_contact():
    cls = _task_class([None], 2)
    install_contact_reset_curriculum(cls, before=2, after=0)
    task = cls()
    try:
        task._reset_ref_state_init(torch.tensor([0]))
    except ValueError as error:
        assert "recorded object contact" in str(error)
    else:
        raise AssertionError("missing contact must fail before running training")
