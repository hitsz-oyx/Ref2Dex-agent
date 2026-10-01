"""Frozen physical predictors for matched return-trained policy features."""
import json
from pathlib import Path
import torch
from scripts.run_contact_response_probe import sha
from scripts.fit_support_response_information import model
from src.task.CmResidual.paired_evaluation import fingerprint
from src.task.CmResidual.support_response import PRIMITIVES
from src.task.CmResidual.support_feature_policy import policy_input

VARIANTS = ('cm','state_only','global_motion_arm')


def evaluation_groups(motion,seed):
    generator=torch.Generator(device='cpu').manual_seed(seed+14000)
    result=torch.full((len(motion),),-1,dtype=torch.long)
    for m in range(3):
        ids=(motion.cpu()==m).nonzero().flatten()
        if len(ids)!=256:raise ValueError('balanced evaluation cohort')
        result[ids]=torch.arange(4).repeat_interleave(64)[torch.randperm(256,generator=generator)]
    return result


class PhysicalFeatureBank:
    def __init__(self, source, device):
        self.source = Path(source)
        self.device = device
        metadata = json.loads((self.source/'results.json').read_text())
        if metadata['run_status'] != 'COMPLETED' or metadata['label'] != 'UNPROMISING':
            raise ValueError('preserve the closed forecast gate')
        self.hashes = {str(self.source/f):sha(self.source/f) for f in
            ('results.json','dataset.pt','fit/cm.pt','fit/state_only.pt','fit/predictions.pt')}
        if sha(self.source/'dataset.pt') != metadata['dataset_sha256']:
            raise ValueError('frozen FIT dataset')
        self.networks = {}
        self.fingerprints = {}
        for variant in ('cm','state_only'):
            path = self.source/'fit'/(variant+'.pt')
            if sha(path) != metadata['fits'][variant]['checkpoint_sha256']:
                raise ValueError('frozen physical predictor')
            payload = torch.load(path,map_location='cpu',weights_only=False)
            network = model().to(device)
            network.load_state_dict(payload['model'])
            network.eval().requires_grad_(False)
            self.networks[variant] = network
            self.fingerprints[variant] = payload['final_model_fingerprint']
            mean,std = payload['mean'].to(device),payload['std'].to(device)
            if variant == 'cm':
                self.mean,self.std = mean,std
            elif not torch.equal(mean,self.mean) or not torch.equal(std,self.std):
                raise ValueError('matched FIT normalization')
        saved = torch.load(self.source/'fit/predictions.pt',map_location='cpu',weights_only=False)
        if sha(self.source/'fit/predictions.pt') != metadata['predictions_sha256']:
            raise ValueError('FIT probabilities provenance')
        self.global_probability = saved['global_fit_probabilities'].to(device,dtype=torch.float32)
        data = torch.load(self.source/'dataset.pt',map_location='cpu',weights_only=False)
        fit = data['seed'] < 525
        if int(fit.sum()) != 4608:
            raise ValueError('frozen FIT rows')
        self.reward_baseline = torch.tensor([data['physical105'][fit & (data['motion']==m)].float().mean()
                                             for m in range(3)],device=device)
        self.macro = torch.tensor(PRIMITIVES,device=device)/torch.tensor([.01,.01,.30],device=device)
        self.assert_frozen()

    @torch.no_grad()
    def features(self, raw, motion):
        n = len(raw)
        if raw.shape != (n,69) or motion.shape != (n,):
            raise ValueError('current physical feature schema')
        z = ((raw-self.mean)/self.std).clamp(-10,10)
        state = z[:,None].expand(-1,8,-1)
        x = torch.cat((state,self.macro[None].expand(n,-1,-1)),-1).reshape(n*8,72)
        zero = torch.cat((state,torch.zeros(n,8,3,device=raw.device)),-1).reshape(n*8,72)
        cm = torch.sigmoid(self.networks['cm'](x)).reshape(n,8)
        state_only = torch.sigmoid(self.networks['state_only'](zero)).reshape(n,8)
        # The global arm executes identical candidate-model work, then discards
        # it and substitutes FIT-only global probabilities. Three equal query
        # budgets; no additional useful model signal enters that control.
        discarded = torch.sigmoid(self.networks['cm'](x)).reshape(n,8)
        if not torch.allclose(discarded,cm,atol=1e-6,rtol=0):
            raise ValueError('matched dummy query')
        probabilities = dict(cm=cm,state_only=state_only,
                             global_motion_arm=self.global_probability[motion])
        inputs = {k:policy_input(z,v) for k,v in probabilities.items()}
        return inputs,probabilities

    def assert_frozen(self):
        if any(sha(Path(p)) != h for p,h in self.hashes.items()):
            raise ValueError('frozen predictor input drift')
        for variant,network in self.networks.items():
            if fingerprint(network.state_dict()) != self.fingerprints[variant]:
                raise ValueError('physical predictor weights changed')
