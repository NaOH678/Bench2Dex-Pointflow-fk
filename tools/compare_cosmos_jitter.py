"""Compare command and measured-joint temporal variation in recorded rollouts."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def variation(values):
    delta = np.diff(values, axis=0)
    second = np.diff(values, n=2, axis=0)
    moving = (abs(delta[:-1]) > .003) & (abs(delta[1:]) > .003)
    return dict(mean_abs_step_rad=float(abs(delta).mean()),
        mean_abs_second_difference_rad=float(abs(second).mean()),
        p95_abs_second_difference_rad=float(np.percentile(abs(second), 95)),
        rms_second_difference_rad=float(np.sqrt(np.mean(second**2))),
        reversal_rate_above_003rad=float((delta[:-1]*delta[1:] < 0)[moving].mean()) if moving.any() else None,
        mean_joint_range_rad=float(np.ptp(values, axis=0).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='LABEL=episode.hdf5')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for item in args.run:
        label, filename = item.split('=', 1)
        with h5py.File(filename) as f:
            q = f['robot/qpos'][:]
            action = f['action/commanded'][:]
            have_names = 'robot/joint_names' in f and 'action/action_names' in f
            q_names = [v.decode() for v in f['robot/joint_names'][:]] if have_names else ['']*q.shape[1]
            a_names = [v.decode() for v in f['action/action_names'][:]] if have_names else ['']*action.shape[1]
            result = dict(label=label, file=filename, frames=len(q), groups={})
            for group in (['all', 'arm', 'finger'] if have_names else ['all']):
                select = lambda names: [i for i, n in enumerate(names) if group == 'all' or ('finger' in n) == (group == 'finger')]
                result['groups'][group] = dict(command=variation(action[:, select(a_names)]),
                                              measured_qpos=variation(q[:, select(q_names)]))
            scene = json.loads(f['meta/scene_generalization_sample'][()])
            result['background'] = scene['appearance']['background']['asset_id']
            result['table'] = scene['appearance']['table_surface']['asset_id']
        results.append(result)
    report = dict(scope='Temporal-variation proxies, not task quality. Lower movement can reduce these metrics; scenes and training iterations must be considered.',
                  runs=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for row in results:
        print(row['label'], row['background'], json.dumps(row['groups']['all']))


if __name__ == '__main__':
    main()
