from threading import Event

import numpy as np
import pytest

from policy.Cosmos.async_actions import AsyncActionQueue


def test_prefetch_overlaps_execution_and_skips_elapsed_actions():
    started, release = Event(), Event()
    source = {'step': 0, 'state': np.array([7.0])}
    calls = []

    def predict(obs):
        calls.append(obs['step'])
        if obs['step'] == 16:
            started.set()
            assert release.wait(5)
            assert obs['state'][0] == 7  # independent observation snapshot
        return (obs['step'] + np.arange(32))[:, None]

    scheduler = AsyncActionQueue(predict)
    try:
        for step in range(32):
            source['step'] = step
            assert scheduler.action(step, lambda: source, episode_steps=64)[0] == step
            if step == 16:
                assert started.wait(5)
                source['state'][0] = 999
        # Reached step32 while the second inference is deliberately unfinished.
        assert calls == [0, 16]
        assert not scheduler.pending[0].done()
        release.set()
        for step in range(32, 64):
            source['step'] = step
            assert scheduler.action(step, lambda: source, episode_steps=64)[0] == step
    finally:
        release.set()
        scheduler.close()
    assert calls == [0, 16, 32]
    assert [e['skipped_actions'] for e in scheduler.events] == [0, 16, 16]
    assert [e['adopt_step'] for e in scheduler.events] == [0, 32, 48]


def test_early_result_keeps_current_chunk_until_exhausted():
    scheduler = AsyncActionQueue(lambda obs: np.full((32, 2), obs['step']))
    try:
        for step in range(48):
            action = scheduler.action(step, lambda: {'step': step}, episode_steps=48)
            np.testing.assert_array_equal(action, [0, 0] if step < 32 else [16, 16])
            if step == 16:
                scheduler.pending[0].result(timeout=5)
    finally:
        scheduler.close()
    assert len(scheduler.events) == 2


@pytest.mark.parametrize('result', [None, np.zeros((16, 52)), np.full((32, 52), np.nan)])
def test_invalid_response_aborts_instead_of_executing(result):
    scheduler = AsyncActionQueue(lambda obs: result)
    try:
        with pytest.raises(ValueError, match='requires 32 finite actions'):
            scheduler.action(0, dict, episode_steps=64)
    finally:
        scheduler.close()


def test_worker_exception_propagates():
    def fail(obs):
        raise RuntimeError('server disconnected')
    scheduler = AsyncActionQueue(fail)
    try:
        with pytest.raises(RuntimeError, match='server disconnected'):
            scheduler.action(0, dict, episode_steps=64)
    finally:
        scheduler.close()
