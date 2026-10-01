import unittest
import torch

from src.task.CmResidual.contact_ranker import (
    frame_group_split,physical_history,shuffled_numeric_actions,
    ConsequenceNetwork,all_predictions,policy_choice,randomized_value,
)


class ContactRankerTests(unittest.TestCase):
    def test_frame_groups_stay_together_across_episodes_and_replicates(self):
        motion=torch.arange(90)%3;frames=torch.arange(90)//3%10
        split,mapping=frame_group_split(motion,frames)
        for i in range(90):self.assertEqual(int(split[i]),mapping[(int(motion[i]),int(frames[i]))])
        self.assertEqual(set(split.tolist()),{0,1,2})
        self.assertTrue(torch.equal(split[:30],split[30:60]))
        for m in range(3):self.assertEqual(set(split[motion==m].tolist()),{0,1,2})

    def test_physical_features_are_invariant_to_world_origin_and_quaternion_sign(self):
        h=torch.randn(7,10,69);h[:,:,39:43]=torch.tensor([0.,0.,0.,1.])
        shifted=h.clone();shifted[:,:,:3]+=2;shifted[:,:,36:39]+=2;shifted[:,:,39:43]*=-1
        original=h.clone();rest=torch.randn(7)
        self.assertTrue(torch.allclose(physical_history(h,rest),physical_history(shifted,rest+2),atol=1e-6))
        self.assertTrue(torch.equal(h,original))

    def test_shuffle_is_per_row_corruption_not_a_global_expert_relabel(self):
        actions=torch.arange(6).float()[None,:,None].expand(64,6,18).clone()
        assignment=torch.arange(64)%6
        selected=shuffled_numeric_actions(actions,assignment)
        corrupt=selected[:,0].long()
        self.assertTrue((corrupt!=assignment).all())
        self.assertGreater(len(set(((corrupt-assignment)%6).tolist())),1)

    def test_numeric_prediction_follows_action_permutation_state_control_does_not(self):
        h=torch.randn(8,10,69);a=torch.randn(8,6,18);context=torch.randn(8,22)
        cm=ConsequenceNetwork();state=ConsequenceNetwork(state_only=True)
        cm.eval();state.eval()
        with torch.no_grad():
            first=all_predictions(cm,h,a,context)
            second=all_predictions(cm,h,a.flip(1),context)
            self.assertTrue(torch.allclose(first.flip(1),second,atol=1e-6))
            self.assertTrue(torch.equal(all_predictions(state,h,a,context),all_predictions(state,h,a.flip(1),context)))
        loss=first.sum()
        self.assertTrue(torch.isfinite(loss))

    def test_uncertain_or_unsupported_lifted_state_keeps_base(self):
        pred=torch.zeros(3,4,6,3);pred[:,:,:,1]=.9;pred[:,:,0,0]=5
        eligible=torch.tensor([False,True,False,True])
        choice,_,_=policy_choice(pred,.5,drop_supported=False,eligible=eligible)
        self.assertEqual(choice.tolist(),[0,4,0,4])
        pred[:,0,0,0]=torch.tensor([-10.,0.,10.])
        choice,_,_=policy_choice(pred,.5,drop_supported=True,eligible=eligible)
        self.assertEqual(int(choice[0]),4)
        pred[:,:,0,1]=.1
        self.assertTrue((policy_choice(pred,.5,drop_supported=True,eligible=eligible)[0]==4).all())

    def test_randomized_value_uses_propensity_not_selected_factual_average(self):
        assignment=torch.arange(60)%6;choice=torch.full_like(assignment,4)
        outcome=torch.full((60,),.3);outcome[assignment!=4]=10
        estimate=randomized_value(choice,assignment,outcome,[f'ep{i//3}' for i in range(60)])
        self.assertAlmostEqual(estimate['value'],.3,places=6)
        self.assertEqual(estimate['matched_windows'],10)
        self.assertEqual(estimate['episodes'],20)


if __name__=='__main__':unittest.main()
