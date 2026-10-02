"""Same independent full expert replay on fresh corrected-goal cohorts."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    s=(ROOT/'scripts/replay_self_trained_teacher.py').read_text().replace('(587,588)','(589,590)')
    exec(compile(s,str(ROOT/'scripts/replay_self_trained_teacher.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/replay_self_trained_teacher.py')})
if __name__=='__main__':main()
