#!/usr/bin/env python3
"""Run the four Task21 channels with persistent progress and bounded crash recovery."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ('none', 'cov_only', 'inv_only', 'inv_cov')


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            break  # A writer may still be appending its final line.
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--anchor-dir', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--reference-run', type=Path, help='Restore the exact recorded baseline scene in each channel')
    a = p.parse_args()
    if a.episodes < 1 or a.episodes > 50:
        p.error('episodes must be between 1 and 50')
    os.chdir(ROOT)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (run / 'supervisor.pid').write_text(str(os.getpid()))
    anchors = a.anchor_dir.resolve()
    config = json.loads(a.config.read_text())
    save(run / 'protocol.json', dict(channels=CHANNELS, episodes_per_channel=a.episodes,
         episode_steps=827, simulation_seconds=41.35, anchor_dir=str(anchors),
         seed_namespace=100000000, inv_cov_seed_namespace=200000000,
         config=config, reference_run=str(a.reference_run.resolve()) if a.reference_run else None,
         started_unix=time.time()))
    status = {'status': 'running', 'counts': {}, 'pid': os.getpid()}

    def update(**kw):
        status.update(kw, heartbeat_unix=time.time())
        save(run / 'progress.json', status)

    try:
        for channel in CHANNELS:
            channel_dir = run / channel
            channel_dir.mkdir(exist_ok=True)
            completed = rows(channel_dir / 'per_episode.jsonl')
            status['counts'][channel] = len(completed)
            update(channel=channel, completed=sum(status['counts'].values()), total=4 * a.episodes)
            gate_path = run / 'asset_gate.json'
            if len(completed) < a.episodes and gate_path.exists():
                gate = json.loads(gate_path.read_text())
                if channel in gate['channels']:
                    while not Path(gate['ready_file']).exists():
                        if Path(gate['failure_file']).exists():
                            raise RuntimeError('Asset preparation failed; see assets.log (no simulation retries consumed)')
                        update(status='waiting_assets', channel=channel)
                        time.sleep(15)
                    update(status='running')
            for attempt in range(1, 4):
                if len(completed) == a.episodes:
                    break
                if [r['episode_index'] for r in completed] != list(range(1, len(completed) + 1)):
                    raise RuntimeError('Non-contiguous episode results; refusing to change seeds')
                start = len(completed) + 1
                folder = channel_dir / f'start_{start:02d}_{time.time_ns()}'
                folder.mkdir()
                cfg = dict(config, output_dir=str(folder / 'model'))
                save(folder / 'deploy.json', cfg)
                env = dict(os.environ, COSMOS_PYTHON=str(ROOT / '.venvs/cosmos-local/bin/python'),
                           COSMOS_CONFIG=str(folder / 'deploy.json'), COSMOS_RUN_DIR=str(folder),
                           BASELINE_ANCHOR_HDF5=str(anchors / 'episode_000000.hdf5'))
                command = ['bash', 'tools/run_cosmos_local_sim.sh', '--seed', '100000000',
                           '--anchor-dir', str(anchors), '--generalization-profile', channel,
                           '--start-episode', str(start), '--num-episodes', str(a.episodes - len(completed)),
                           '--episode-steps', '827', '--early-stop']
                if config.get('backend') == 'pointfk_bundle':
                    command += ['--live-pointfk']
                if a.reference_run:
                    command += ['--reference-episodes', str(a.reference_run.resolve() / channel / 'per_episode.jsonl')]
                catalog_config = run / 'generalization_full_catalog.json'
                if channel in ('inv_only', 'inv_cov'):
                    if not catalog_config.exists():
                        raise RuntimeError('Missing pinned full-catalog generalization config')
                    command += ['--generalization-config', str(catalog_config)]
                save(folder / 'command.json', command)
                previous = list(completed)
                with (folder / 'launch.log').open('w') as log:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    while True:
                        new = rows(folder / 'metrics/per_episode.jsonl')
                        completed = previous + new
                        target = channel_dir / 'per_episode.jsonl'
                        tmp = target.with_suffix('.tmp')
                        tmp.write_text(''.join(json.dumps(r) + '\n' for r in completed))
                        tmp.replace(target)
                        status['counts'][channel] = len(completed)
                        update(channel=channel, attempt=attempt, child_pid=child.pid,
                               current_dir=str(folder), completed=sum(status['counts'].values()),
                               total=4 * a.episodes)
                        if child.poll() is not None:
                            # Re-read after process exit on the next pass before leaving.
                            final = rows(folder / 'metrics/per_episode.jsonl')
                            if len(final) != len(new):
                                continue
                            break
                        time.sleep(15)
                save(folder / 'exit_status.json', dict(returncode=child.returncode, time=time.time()))
                if any(r.get('error') for r in completed):
                    raise RuntimeError(f'{channel}: infrastructure error in episode metrics; inspect logs')
                if len(completed) == a.episodes:
                    break
                print(f'{channel}: exit={child.returncode}; completed={len(completed)}; retry {attempt}/3', flush=True)
            if len(completed) != a.episodes:
                raise RuntimeError(f'{channel}: incomplete after 3 attempts')
        with (run / 'summary.log').open('w') as log:
            subprocess.run([str(ROOT / '.venvs/cosmos-local/bin/python'),
                            'tools/compute_robustness.py', str(run), '--expected-n', str(a.episodes)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        update(status='complete', finished_unix=time.time())
    except BaseException as exc:
        update(status='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
