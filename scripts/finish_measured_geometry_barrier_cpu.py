"""Resource-safe fit-only completion using no occupied GPU and no native repetition."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=(ROOT/'scripts/finish_measured_geometry_barrier_fit.py').read_text()
    changes={
        "ROOT/'scripts/finish_measured_geometry_barrier_fit.py',":"ROOT/'scripts/finish_measured_geometry_barrier_fit.py',ROOT/'scripts/finish_measured_geometry_barrier_cpu.py',ROOT/'scripts/fit_measured_geometry_barriers_cpu.py',ROOT/'docs/archive/2026-10-04-research-governance/decisions/D-20261002-geometry-cpu-fit-fallback.md',",
        "verify();gpu=admission(a.gpu_index);out.mkdir();begin=time.monotonic()":"verify();gpu=dict(index=None,uuid='',execution_device='cpu',reason='all eight GPU devices occupied; matched common CPU batch stream exception');out.mkdir();begin=time.monotonic()",
        "    def guard():\n":"    def guard():\n",
        "str(ROOT/'scripts/fit_measured_geometry_barriers.py')":"str(ROOT/'scripts/fit_measured_geometry_barriers_cpu.py')",
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('frozen fit-only parent drift '+old)
        source=source.replace(old,new)
    exec(compile(source,str(ROOT/'scripts/finish_measured_geometry_barrier_fit.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/finish_measured_geometry_barrier_fit.py')})

if __name__=='__main__':main()
