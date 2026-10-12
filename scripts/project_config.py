"""Resolve portable project paths and parameters for each workflow stage."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads(Path(os.environ.get('TC_CONFIG', ROOT / 'configs/paper.json')).read_text(encoding='utf-8'))
SAMPLE = os.environ.get('TC_SAMPLE', 'LHJ2')
RUN_DIR = Path(os.environ.get('TC_RUN_DIR', ROOT / 'results/default')).resolve()
PROCESSED_DIR = RUN_DIR / 'processed'
MODEL_DIR = Path(os.environ.get('TC_MODEL_DIR', ROOT / 'models/reference')).resolve()
STAGE = os.environ.get('TC_STAGE', 'prepare')
STAGE_OUTPUT = RUN_DIR / STAGE
STAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
DATA_FILE = PROCESSED_DIR / (SAMPLE + '.xlsx')
FEATURES = CFG['features']
FEATURE_INDICES_CONFIG = list(range(1, len(FEATURES) + 1))
