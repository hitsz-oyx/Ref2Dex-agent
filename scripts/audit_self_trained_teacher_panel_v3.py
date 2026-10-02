"""Launch corrected audit from the native subprocess working directory."""
import runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if __name__=='__main__':runpy.run_module('scripts.audit_self_trained_teacher_panel_v2',run_name='__main__')
