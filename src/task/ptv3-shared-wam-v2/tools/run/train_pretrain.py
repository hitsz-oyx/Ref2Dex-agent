"""Independent v2 entry point; accepts single GPU or torchrun."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ptv3_shared_wam_v2.training import run

if __name__ == "__main__":
    run()
