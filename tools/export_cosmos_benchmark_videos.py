"""Export completed benchmark episodes, optionally following an active run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run_dir', type=Path)
p.add_argument('--follow', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
run = a.run_dir.resolve()
lock = (run / 'video_export.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
(run / 'video_export.pid').write_text(str(os.getpid()))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
while True:
    for channel in ('none', 'cov_only', 'inv_only', 'inv_cov'):
        metrics = run / channel / 'per_episode.jsonl'
        if not metrics.exists():
            continue
        records = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        records.sort(key=lambda row: not row['success'])
        for row in records:
            label = 'success' if row['success'] else 'failure'
            name = f"episode_{row['episode_index']:06d}"
            target = run / 'videos' / channel / label / (name + '.mp4')
            if target.exists():
                continue
            sources = list((run / channel).glob(f'*/episodes/{label}/{name}.hdf5'))
            if len(sources) != 1:
                raise RuntimeError(f'Expected one completed recording for {channel}/{name}, found {sources}')
            partial = target.with_name(name + '.partial.mp4')
            subprocess.run([sys.executable, str(root / 'tools/render_cosmos_episode.py'),
                            '--episode', str(sources[0]), '--output', str(partial),
                            '--three-views', '--fps', '20', '--threads', '2'], check=True)
            partial.replace(target)
            print('READY', target, flush=True)
    status = json.loads((run / 'progress.json').read_text())['status']
    if not a.follow or status in ('complete', 'failed'):
        break
    time.sleep(20)
