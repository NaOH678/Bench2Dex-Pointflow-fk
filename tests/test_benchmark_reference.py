import json
import pytest
from utils.benchmark_reference import reference_episode, verify_reference_scene


def test_reference_preserves_scene_and_rejects_wrong_seed(tmp_path):
    sample = dict(resolved_object_placements={'box': {'position': [1, 2, 3]}}, appearance={'background': 'a'})
    p = tmp_path / 'episodes.jsonl'
    p.write_text(json.dumps(dict(episode_index=2, seed=123, perturbation_axis='none',
                                error=None, scene_generalization_sample=sample))+'\n')
    result = reference_episode(p, 2, 123, 'none')
    verify_reference_scene(sample, result)
    result['appearance']['background'] = 'b'
    with pytest.raises(ValueError, match='appearance'):
        verify_reference_scene(sample, result)
    with pytest.raises(ValueError, match='seed/profile'):
        reference_episode(p, 2, 124, 'none')
    with pytest.raises(ValueError, match='one reference'):
        reference_episode(p, 1, 123, 'none')
