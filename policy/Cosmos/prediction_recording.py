"""Save metric joint predictions for offline visualization; no rendering here."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def save_prediction(path, geometry, predictions, actions, *, seed, query_index):
    started = perf_counter()
    meta = geometry['metadata']
    point = geometry['pointflow_inputs']
    width, height = map(int, point['image_size_wh'])
    intrinsic = np.diag([width, height, 1]) @ point['intrinsics_normalized']
    metadata = dict(schema='cosmos_live_pointfk_prediction_v1',
                    frame_id=int(meta['frame_id']), sim_time_sec=float(meta['sim_time_sec']),
                    fps=20, horizon=32, seed=int(seed), query_index=int(query_index),
                    coordinate=meta['coordinate'], units='metre',
                    representation='displacement_from_current_anchor_not_frame_increment',
                    executed_action_count=len(actions), uses_future_labels=False,
                    point_identity_scope='within_this_query_only')
    arrays = dict(metadata_json=np.array(json.dumps(metadata)), intrinsic=intrinsic,
                  image_size_wh=np.array([width, height]), executed_actions=np.asarray(actions))
    for name in ('pointflow', 'fk'):
        inputs = geometry[name + '_inputs']
        displacement = np.asarray(predictions[name], np.float32)
        anchor = np.asarray(inputs['anchor_xyz'], np.float32)
        if displacement.shape != (32, len(anchor), 3) or not np.isfinite(displacement).all():
            raise ValueError(f'Invalid {name} prediction archive')
        arrays[name + '_anchor_xyz'] = anchor
        arrays[name + '_anchor_uv'] = np.asarray(inputs['anchor_uv'], np.float32)
        arrays[name + '_point_ids'] = np.asarray(inputs['point_ids'])
        # The formal sampler already returned metres: do not rescale or cumsum.
        arrays[name + '_displacement'] = displacement
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed NPZ avoids compression work on the control path.
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
    return perf_counter() - started
