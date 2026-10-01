import unittest
import tempfile
from pathlib import Path
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

    def test_frozen_runtime_matches_training_inputs_and_preserves_actor_rng(self):
        from src.task.CmResidual.contact_selector import FrozenContactSelector
        models=[ConsequenceNetwork().eval() for _ in range(3)]
        h=torch.randn(8,10,69);a=torch.randn(8,6,18);rest=torch.zeros(8)
        motion=torch.arange(8)%3;start=torch.arange(8);trigger=torch.full_like(start,20)
        saved=dict(schema='ref2dex.contact_ranker.v1',models={'cm':[m.state_dict() for m in models]},
                   history_mean=torch.randn(69),history_scale=torch.rand(69)+1,
                   action_mean=torch.randn(18),action_scale=torch.rand(18)+1,
                   context_mean=torch.randn(22),context_scale=torch.rand(22)+1,
                   lift_mean=torch.tensor(3.),lift_scale=torch.tensor(5.),drop_supported=True,
                   calibration={'cm':{'margin_mm':.5}})
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'model.pt';torch.save(saved,checkpoint)
            before=torch.get_rng_state();selector=FrozenContactSelector(checkpoint,'cpu')
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            result=selector.predict(h,a,rest,motion,start,trigger)
        context=torch.cat((a[:,4],torch.nn.functional.one_hot(motion,3).float(),((start+trigger)/600)[:,None]),-1)
        context=(context-saved['context_mean'])/saved['context_scale']
        predictions=[]
        with torch.no_grad():
            for model in models:
                raw=all_predictions(model,(physical_history(h,rest)-saved['history_mean'])/saved['history_scale'],
                                    (a-saved['action_mean'])/saved['action_scale'],context)
                predictions.append(torch.stack(((raw[:,:,0]*5+3).clamp_min(0),raw[:,:,1].sigmoid(),raw[:,:,2].sigmoid()),-1))
        choice,mean,lower=policy_choice(torch.stack(predictions),.5,drop_supported=True,eligible=h[:,-1,38]-rest>=.03)
        self.assertTrue(torch.equal(choice,result['proposed_arm']))
        self.assertTrue(torch.allclose(result['lower_gain_mm'],lower[torch.arange(8),choice]))

    def test_targeted_contrast_uses_known_allocation_and_clusters_same_records(self):
        from scripts.analyze_targeted_contact import contrast
        treatment=torch.tensor([True,False]*30)
        outcome=torch.where(treatment,3.,1.)
        clusters=[f'episode{i//2}' for i in range(60)]
        result=contrast(treatment,outcome,clusters,clusters)
        self.assertAlmostEqual(result['effect'],2.)
        self.assertEqual(result['frame_cluster90'],[2.,2.])
        self.assertEqual(result['treated_rows'],30)
        # Known-propensity IPW retains allocation noise; do not substitute an
        # observational arm-mean difference when counts are unequal.
        result=contrast(torch.tensor([True,True,False]),torch.tensor([3.,3.,1.]),['a','b','c'],['a','b','c'])
        self.assertAlmostEqual(result['effect'],10/3)


if __name__=='__main__':unittest.main()
