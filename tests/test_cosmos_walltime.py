import numpy as np
import pytest

from tools.render_cosmos_walltime import timeline, phase_at


def test_async_rpc_is_not_all_labeled_as_paused_physics():
    timing = dict(async_inference=True, queries=[dict(geometry_start_sec=1.,
                  geometry_end_sec=1.1, rpc_end_sec=3., queue_wait_sec=.4)])
    assert phase_at(timing, 1.05) == 'WAIT: POINT + FK PREPARATION'
    assert phase_at(timing, 2.) == 'ASYNC MODEL + RPC / SIM CONTINUES'
    assert phase_at(timing, 2.8) == 'WAIT: EMPTY ACTION QUEUE / MODEL + RPC'
    assert phase_at(timing, 3.1) == 'SIM / CAMERA / RECORDING'
    timing['queries'][0]['queue_wait_sec'] = .000001
    assert phase_at(timing, 2.999) == 'ASYNC MODEL + RPC / SIM CONTINUES'
    timing['async_inference'] = False
    assert phase_at(timing, 2.) == 'WAIT: MODEL + RPC (PHYSICS PAUSED)'


def test_walltime_keeps_initial_inference_wait_and_last_frame():
    timing = dict(record_stride=1, frames=[dict(captured_wall_sec=t) for t in [.1, 2.1, 2.2]],
                  end_wall_sec=2.3)
    clock, indices = timeline(timing, 20)
    assert len(indices) == 44
    np.testing.assert_array_equal(indices[:40], np.zeros(40, dtype=int))
    assert indices[40] == 1 and indices[-1] == 2
    assert clock[0] == .1  # Startup excluded, never synthesize an unobserved frame.


@pytest.mark.parametrize('times,end,stride', [([1., 1.], 2., 1), ([1., 2.], 1.5, 1), ([1.], 2., 3)])
def test_invalid_or_misaligned_timing_rejected(times, end, stride):
    with pytest.raises(ValueError):
        timeline(dict(record_stride=stride, frames=[dict(captured_wall_sec=t) for t in times],
                      end_wall_sec=end), 20)
