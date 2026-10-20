#!/usr/bin/env python3
"""Collect a bounded UniDexGrasp++ live-state oracle-tau identification probe."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

DEFAULT_OBJECT = 'core/bottle-109d55a137c042f5760315ac3bf2c13e'
REPLAY = Path(os.environ.get('REF2DEX_RUN_ROOT',Path(__file__).resolve().parents[5])) / 'src/task/video-data-preparation/tools/run/replay_unidexgrasp.py'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, indent=2))
    partial.replace(path)


def load_replay_helpers():
    spec = importlib.util.spec_from_file_location('replay_unidexgrasp', REPLAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
