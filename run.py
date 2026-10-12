"""Command-line entry point; every run records parameters and its actual environment."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
STAGES = {'prepare': 'prepare_data.py', 'evaluate': 'evaluate_models.py',
          'train': 'train_bilstm.py', 'rf': 'train_random_forest.py',
          'shap': 'explain_shap.py', 'predict': 'predict_fossil.py',
          'events': 'extract_events.py', 'domains': 'compare_domains.py',
          'features': 'plot_features.py'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=list(STAGES) + ['reproduce'])
    parser.add_argument('--sample', choices=['XB10', 'LHJ2'], default='LHJ2')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/paper.json')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'results/default')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/reference')
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == 'train' and args.run_dir == (ROOT / 'models/reference').resolve():
        parser.error('Choose a results directory to preserve reference checkpoints.')
    stages = ['prepare', 'evaluate', 'predict', 'events', 'domains', 'features'] if args.stage == 'reproduce' else [args.stage]
    env = os.environ.copy()
    env.update(TC_CONFIG=str(args.config.resolve()), TC_RUN_DIR=str(args.run_dir),
               TC_MODEL_DIR=str(args.model_dir.resolve()), MPLBACKEND='Agg', PYTHONUTF8='1',
               PYTHONDONTWRITEBYTECODE='1')
    versions = {}
    for package in ['numpy','pandas','scipy','scikit-learn','torch','shap','matplotlib','seaborn','joblib','openpyxl']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    metadata = {'started_utc':stamp, 'stage':args.stage, 'sample':args.sample,
                'python':sys.version, 'platform':platform.platform(), 'packages':versions,
                'configuration':json.loads(args.config.read_text(encoding='utf-8')),
                'model_dir':str(args.model_dir.resolve()), 'completed_stages':[]}
    logdir = args.run_dir / 'logs'
    logdir.mkdir(exist_ok=True)
    meta_path = logdir / (stamp + '_manifest.json')
    for stage in stages:
        env.update(TC_STAGE=stage, TC_SAMPLE='A114' if stage == 'predict' else args.sample)
        with (logdir / (stamp + '_' + stage + '.log')).open('w', encoding='utf-8') as log:
            print('Running', stage, flush=True)
            result = subprocess.run([sys.executable, str(ROOT/'scripts'/STAGES[stage])], env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            metadata['failed_stage'] = stage
            meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            print('Stage failed. See', log.name, file=sys.stderr)
            return result.returncode
        metadata['completed_stages'].append(stage)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Completed. Outputs:', args.run_dir)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
