"""Pin evaluation to recorded baseline scenes, independent of current sampling order."""
import copy
import json
from pathlib import Path


def reference_episode(path, episode_index, seed, profile):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    matches = [r for r in rows if r['episode_index'] == episode_index]
    if len(matches) != 1:
        raise ValueError(f'Expected one reference episode {episode_index}')
    row = matches[0]
    axis = {'none': 'none', 'cov_only': 'cov', 'inv_only': 'inv', 'inv_cov': 'inv+cov'}[profile]
    if row['seed'] != seed or row['perturbation_axis'] != axis or row.get('error'):
        raise ValueError('Reference seed/profile/error does not match evaluation')
    sample = row['scene_generalization_sample']
    if not sample.get('resolved_object_placements'):
        raise ValueError('Reference lacks resolved object placements')
    return copy.deepcopy(sample)


def verify_reference_scene(expected, actual):
    differences = [key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key)]
    if differences:
        raise ValueError(f'Built scene differs from baseline reference: {sorted(differences)}')
