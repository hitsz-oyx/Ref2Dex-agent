"""Frozen matched forecasts of a NEW executed four-tick request contract."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build_source():
    source=(ROOT/'scripts/fit_measured_geometry_barriers.py').read_text();start=source.index('def read_panel(');end=source.index('\ndef features(',start);source=source[:start]+source[end:]
    changes={
        'from src.task.CmResidual.measured_geometry_barriers import SCHEMA,relative_geometry,initialized_model':'from src.task.CmResidual.measured_geometry_barriers import initialized_model\nfrom src.task.CmResidual.coherent_barrier_information import SCHEMA,read_panel',
        "p.add_argument('--source',type=Path,required=True);":"p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--source',type=Path,required=True);",
        'assert torch.cuda.is_available();':'device=a.device;assert device=="cpu" or torch.cuda.is_available();',
        "(578,579)":"(584,585)","a.source/'s580'":"a.source/'s586'",
        "ROOT/'docs/experiments/probes/P-20261002-measured-geometry-barriers.md'":"ROOT/'docs/experiments/probes/P-20261002-coherent-barrier-information.md',ROOT/'scripts/fit_measured_geometry_barriers.py',ROOT/'src/task/CmResidual/coherent_barrier_information.py'",
        "initialized_model().cuda()":"initialized_model(3311).cuda()",
        "torch.Generator(device='cuda').manual_seed(3212)":"torch.Generator(device='cpu').manual_seed(3312)",
        "device='cuda',generator=gen)":"device='cpu',generator=gen).to(device)",
        'np.random.RandomState(3213)':'np.random.RandomState(3313)',
        "fit_only_statistics=True,":"fit_only_statistics=True,execution_device=device,batch_rng_device='cpu',coherent_period=4,target_horizon_ticks=4,",
        "actual_sdk_measured_poses=True,":"actual_sdk_measured_poses=True,execution_device=device,batch_rng_device='cpu',coherent_period=4,target_horizon_ticks=4,",
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('exact coherent fit marker drift '+old+':'+str(source.count(old)))
        source=source.replace(old,new)
    if source.count('.cuda()')!=4:raise ValueError('fixed model device conversion')
    return source.replace('.cuda()', '.to(device)')

def main():
    exec(compile(build_source(),str(ROOT/'scripts/fit_measured_geometry_barriers.py'),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})

if __name__=='__main__':main()
