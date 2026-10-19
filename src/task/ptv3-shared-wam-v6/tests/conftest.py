from pathlib import Path
import sys
TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK/'src'))
sys.path.insert(0,str(TASK.parent/'ptv3-shared-wam-v5/tests'))
