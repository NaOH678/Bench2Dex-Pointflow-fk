import json

import numpy as np
import pytest

from policy.Cosmos.prediction_recording import save_prediction
from tools.render_cosmos_pointfk_predictions import positions, project, query_for_frame


def test_metric_predictions_roundtrip_without_accumulation_or_rescale(tmp_path):
    geometry = dict(metadata=dict(frame_id=48, sim_time_sec=.8, coordinate='fixed_optical_camera'))
    predictions = {}
    for name, count in [('pointflow', 5), ('fk', 42)]:
        geometry[name + '_inputs'] = dict(anchor_xyz=np.tile([0., 0., 1.], (count, 1)),
                                          anchor_uv=np.tile([320., 240.], (count, 1)),
                                          point_ids=np.arange(count))
        predictions[name] = np.tile([.1, 0., 0.], (32, count, 1))
    geometry['pointflow_inputs'].update(image_size_wh=np.array([640, 480]),
        intrinsics_normalized=np.diag([1 / 640, 1 / 480, 1]) @ np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]))
    path = tmp_path / 'query.npz'
    save_prediction(path, geometry, predictions, np.zeros((16, 52)), seed=42, query_index=1)
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data['metadata_json']))['frame_id'] == 48
        for name in predictions:
            np.testing.assert_allclose(positions(data, name, 32), positions(data, name, 1))
        uv, valid = project(positions(data, 'pointflow', 32), data['intrinsic'], 640, 480)
        np.testing.assert_array_equal(uv, np.tile([370, 240], (5, 1)))
        assert valid.all()
    with pytest.raises(FileExistsError):
        save_prediction(path, geometry, predictions, np.zeros((16, 52)), seed=42, query_index=1)


def test_replan_boundaries_and_invalid_projection():
    starts = np.array([0, 16, 32])
    assert query_for_frame(starts, 15) == (0, 15)
    assert query_for_frame(starts, 16) == (1, 0)
    assert query_for_frame(starts, 17) == (1, 1)
    with pytest.raises(ValueError):
        query_for_frame(starts, 65)
    _, valid = project(np.array([[1, 1, 1], [1, 1, -1], [np.nan, 1, 1], [900, 1, 1]]),
                       np.eye(3), 640, 480)
    assert valid.tolist() == [True, False, False, False]


def test_async_overlay_uses_adopted_prediction_and_skips_expired_prefix():
    starts = np.array([0, 16, 32, 48])
    adopted = [0, 32, 48, None]
    assert query_for_frame(starts, 16, adopted) == (0, 16)
    assert query_for_frame(starts, 31, adopted) == (0, 31)
    assert query_for_frame(starts, 32, adopted) == (1, 16)
    assert query_for_frame(starts, 49, adopted) == (2, 17)
