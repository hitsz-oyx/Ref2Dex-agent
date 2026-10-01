import unittest
import torch

from src.task.CmResidual.contact_trajectory import (
    trajectory_targets,TrajectoryNetwork,all_trajectories,decode_trajectories,retained_choice,
)


class ContactTrajectoryTests(unittest.TestCase):
    def test_retained_progress_rejects_transient_height_and_incomplete_contact(self):
        future=torch.zeros(4,10,49);contact=torch.ones(4,10,dtype=torch.bool)
        future[:,:,38]=1.04;future[1,-3:,38]=1.;contact[2,-1]=False
        trigger=torch.ones(4);rest=torch.ones(4)
        result=trajectory_targets(future,contact,trigger,rest)
        self.assertAlmostEqual(float(result['retained_lift_mm'][0]),40,places=3)
        self.assertEqual(float(result['retained_lift_mm'][1]),0.)
        self.assertEqual(float(result['retained_lift_mm'][2]),0.)
        self.assertTrue(bool(result['release'][1]))
        self.assertFalse(bool(result['initially_lifted'][1]))
        # All-window acquisition-then-release labels do not select a subset
        # based on a future consequence of the randomized action.
        self.assertEqual(result['release'].shape,(4,))

    def test_contact_loss_before_first_lift_does_not_count_as_post_lift_release(self):
        f=torch.zeros(2,10,49);f[:,:,38]=1.;f[:,5:,38]=1.04
        contact=torch.zeros(2,10,dtype=torch.bool);contact[0,6:]=True
        result=trajectory_targets(f,contact,torch.ones(2),torch.ones(2))
        self.assertFalse(result['release'].bool().any())
        f[1,4:,38]=1.04
        self.assertTrue(bool(trajectory_targets(f,contact,torch.ones(2),torch.ones(2))['release'][1]))

    def test_base_effect_is_exactly_zero_after_nonzero_weight_changes(self):
        model=TrajectoryNetwork().eval();h=torch.randn(7,10,69);a=torch.randn(7,18);c=torch.randn(7,22)
        with torch.no_grad():model.effect_head[-1].weight.normal_();model.effect_head[-1].bias.normal_()
        baseline,effect=model.components(h,a,a,c)
        self.assertTrue(torch.equal(effect,torch.zeros_like(effect)))
        self.assertTrue(torch.equal(model(h,a,a,c),baseline))

    def test_numeric_trajectory_permutation_and_state_control_pool_invariance(self):
        h=torch.randn(4,10,69);a=torch.randn(4,6,18);c=torch.randn(4,22)
        model=TrajectoryNetwork().eval();state=TrajectoryNetwork(state_only=True).eval()
        with torch.no_grad():model.effect_head[-1].weight.normal_()
        perm=torch.tensor([1,0,3,2,4,5]) # keep the designated numeric base fixed
        self.assertTrue(torch.allclose(all_trajectories(model,h,a[:,perm],c),all_trajectories(model,h,a,c)[:,perm],atol=1e-5))
        self.assertTrue(torch.equal(all_trajectories(state,h,a,c),all_trajectories(state,h,a[:,perm],c)))

    def test_release_uncertainty_and_joint_contact_guard_retained_choice(self):
        raw=torch.zeros(3,5,6,22);raw[:,:,:,20]=4;raw[:,:,:,21]=-4
        raw[:,:,0,:10]=10
        pred=decode_trajectories(raw,torch.zeros(10),torch.ones(10))
        choice,_=retained_choice(pred,.5,True);self.assertEqual(choice.tolist(),[0]*5)
        raw[:,:,0,21]=-3
        pred=decode_trajectories(raw,torch.zeros(10),torch.ones(10))
        self.assertEqual(retained_choice(pred,.5,True)[0].tolist(),[4]*5)
        raw[:,:,0,21]=-4;raw[:,:,0,20]=0
        self.assertEqual(retained_choice(decode_trajectories(raw,torch.zeros(10),torch.ones(10)),.5,True)[0].tolist(),[4]*5)
        self.assertEqual(retained_choice(pred,.5,False)[0].tolist(),[4]*5)


if __name__=='__main__':unittest.main()
