import json
import numpy as np
import pytest

from policy.Cosmos.action_smoothing import ActionSmoother
from policy.Cosmos import deploy_policy


def test_disabled_is_exact_passthrough():
    raw = np.array([[1., -2.]])
    assert ActionSmoother().apply(raw, None, None) is raw


def test_chunk_boundaries_equal_single_continuous_filter_and_reset():
    raw = np.array([[1., -1.], [-1., 1.], [1., -1.], [-1., 1.]], np.float32)
    names = ['a', 'b']
    whole = ActionSmoother({'mode': 'ema', 'alpha': .3}).apply(raw, [0, 0], names)
    filt = ActionSmoother({'mode': 'ema', 'alpha': .3})
    first = filt.apply(raw[:2], [0, 0], names)
    second = filt.apply(raw[2:], [9, 9], names)
    np.testing.assert_array_equal(np.concatenate([first, second]), whole)
    assert np.abs(np.diff(whole, n=2, axis=0)).mean() < np.abs(np.diff(raw, n=2, axis=0)).mean()
    filt.reset()
    np.testing.assert_allclose(filt.apply([[2, 2]], [1, 1], names), [[1.3, 1.3]])
    np.testing.assert_array_equal(raw[0], [1, -1])


@pytest.mark.parametrize('alpha', [0, -1, 1.01, float('nan')])
def test_reject_invalid_alpha(alpha):
    with pytest.raises(ValueError):ActionSmoother({'mode': 'ema', 'alpha': alpha})


def test_joint_order_change_rejected_without_corrupting_state():
    f = ActionSmoother({'mode': 'ema'})
    f.apply([[1, 2]], [0, 0], ['a', 'b'])
    before = f.previous.copy()
    with pytest.raises(ValueError):f.apply([[1, 2]], [0, 0], ['b', 'a'])
    np.testing.assert_array_equal(f.previous, before)


def test_session_filters_runtime_radians_and_resets(monkeypatch, tmp_path):
    class Model:
        def reset(self, seed):pass
        def get_action(self, observation):return np.ones((4, 2), np.float32)
    monkeypatch.setattr(deploy_policy, 'get_model', lambda config: Model())
    session = deploy_policy.LocalSession({'output_dir': str(tmp_path), 'action_smoothing': {'mode': 'ema', 'alpha': .5}})
    obs = {'joint_names': ['b', 'a'], 'joint_action': {'vector': [0, 0]}}
    np.testing.assert_allclose(session.get_action_chunk(obs)[:, 0], [.5, .75, .875, .9375])
    assert session.get_action_chunk(obs)[0, 0] == .96875
    session.reset(seed=42)
    assert session.get_action_chunk(obs)[0, 0] == .5
    records=[json.loads(x) for x in (tmp_path/'action_smoothing.jsonl').read_text().splitlines()]
    assert records[0]['joint_names']==['b', 'a']
    assert records[0]['raw'][0]==[1, 1]
