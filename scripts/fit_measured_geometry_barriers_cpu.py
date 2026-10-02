"""Resource exception: same frozen scientific fit with a common CPU batch stream."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=(ROOT/'scripts/fit_measured_geometry_barriers.py').read_text()
    for old,new,expected in [("assert torch.cuda.is_available()","assert not torch.cuda.is_available()",1),(".cuda()",".to('cpu')",4),("torch.Generator(device='cuda')","torch.Generator(device='cpu')",1),("device='cuda',generator=gen","device='cpu',generator=gen",1)]:
        if source.count(old)!=expected:raise ValueError('fixed CPU conversion marker drift '+old+':'+str(source.count(old)))
        source=source.replace(old,new)
    exec(compile(source,str(ROOT/'scripts/fit_measured_geometry_barriers.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/fit_measured_geometry_barriers.py')})

if __name__=='__main__':main()
