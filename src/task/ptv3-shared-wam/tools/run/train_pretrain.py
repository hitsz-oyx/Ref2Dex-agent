"""Main-three pretraining; use --smoke for a four-update engineering check."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ptv3_shared_wam.training import run

if __name__ == "__main__":
    run("pretrain")
