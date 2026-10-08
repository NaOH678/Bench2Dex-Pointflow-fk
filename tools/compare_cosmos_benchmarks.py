"""Compare paired four-channel benchmarks, including LSCR and exact scene checks."""
import argparse
import json
from pathlib import Path
import time

CHANNELS = ('none', 'cov_only', 'inv_only', 'inv_cov')


def read_rows(root, channel):
    path = root / channel / 'per_episode.jsonl'
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def metrics(rows):
    n = len(rows)
    return dict(n=n, successes=sum(bool(r['success']) for r in rows),
                sr=sum(bool(r['success']) for r in rows)/n if n else None,
                lscr=sum(r['latched_stage_completion_rate'] for r in rows)/n if n else None)


def compare(baseline, candidate):
    result = dict(baseline=str(baseline), candidate=str(candidate), complete=True, paired_scenes_valid=True, channels={})
    all_base, all_new = [], []
    for channel in CHANNELS:
        old, new = read_rows(baseline, channel), read_rows(candidate, channel)
        index = {r['episode_index']: r for r in old}
        problems = []
        for row in new:
            prior = index.get(row['episode_index'])
            if prior is None or prior['seed'] != row['seed'] or prior['scene_generalization_sample'] != row['scene_generalization_sample']:
                problems.append(dict(episode_index=row['episode_index'], problem='seed or scene differs'))
            if row.get('error'):
                problems.append(dict(episode_index=row['episode_index'], problem=row['error']))
        ids = [r['episode_index'] for r in new]
        if len(set(ids)) != len(ids):
            problems.append(dict(problem='duplicate episode indices'))
        complete = set(ids) == set(index) and len(ids) == len(old) and not problems
        result['complete'] &= complete
        result['paired_scenes_valid'] &= not problems
        result['channels'][channel] = dict(baseline=metrics(old), candidate=metrics(new), complete=complete,
                                           problems=problems, missing=sorted(set(index)-set(ids)))
        all_base.extend(old); all_new.extend(new)
    result['overall'] = dict(baseline=metrics(all_base), candidate=metrics(all_new))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--follow', action='store_true', help='Update paired metrics until supervisor finishes')
    a = p.parse_args()
    previous_n = -1
    while True:
        result = compare(a.baseline, a.candidate)
        path = a.candidate / 'comparison.json'
        tmp = path.with_suffix('.compare_tmp'); tmp.write_text(json.dumps(result, indent=2)+'\n'); tmp.replace(path)
        n = result['overall']['candidate']['n']
        if not a.follow or n != previous_n:
            print(json.dumps(result, indent=2), flush=True)
            previous_n = n
        status = json.loads((a.candidate / 'progress.json').read_text())['status'] if a.follow else None
        if not a.follow or status in ('complete', 'failed'):
            break
        time.sleep(15)


if __name__ == '__main__':
    main()
