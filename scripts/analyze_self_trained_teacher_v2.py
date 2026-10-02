"""Same preregistered gates, corrected holding-goal data only."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    s=(ROOT/'scripts/analyze_self_trained_teacher.py').read_text().replace('(587,588)','(589,590)')
    s=s.replace("audit['run_status']==replay['run_status']=='COMPLETED'","audit['run_status']==replay['run_status']=='COMPLETED' and audit['reference_goal_contract_independently_rebuilt']")
    exec(compile(s,str(ROOT/'scripts/analyze_self_trained_teacher.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/analyze_self_trained_teacher.py')})
if __name__=='__main__':main()
