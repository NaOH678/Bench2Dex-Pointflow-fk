"""Current-observation Point/FK inputs for the training source's joint sampler."""
import json
import os
from pathlib import Path
import threading

import numpy as np

from policy.Cosmos.bundle_policy import BundlePolicy

TRAINING_VIEW_DESCRIPTION = 'Head on top; left wrist bottom-left; right wrist bottom-right.'


def align_training_prompt(batch, video_key, task):
    """Use the Point/FK dataset's exact formatter inputs, not the old baseline text."""
    from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
    data = dict(ai_caption=task,video=batch[video_key][0][0],action=batch['action'][0][0],
                conditioning_fps=batch['conditioning_fps'][0],image_size=batch['image_size'][0],
                mode='wam',viewpoint='concat_view',additional_view_description=TRAINING_VIEW_DESCRIPTION)
    prompt = json.dumps(ActionPromptJsonFormatter()(data)['ai_caption'])
    batch['ai_caption'] = [prompt]
    batch['prompt'] = [prompt]
    return batch


def geometry_samples(geometry, image_size):
    """Create unlabeled payloads; no future point/FK coordinates are accepted."""
    from cosmos_framework.data.pointflow_window import PointFlowTiming
    from cosmos_framework.data.fk_window import FKTiming
    from cosmos_framework.data.generator.action.pointflow_source import resize_pointflow_metadata

    meta = geometry['metadata']
    if meta['uses_future_frames'] or meta['coordinate'] != 'fixed_optical_camera' or meta['units'] != 'metre':
        raise ValueError('Expected current camera-frame metric geometry')
    result = []
    for key, timing, expected in [('pointflow_inputs', PointFlowTiming(20.,32,4),1024),
                                  ('fk_inputs', FKTiming(20.,32,4),42)]:
        inputs = {k: np.array(v, copy=True) for k,v in geometry[key].items()}
        n = len(inputs['point_ids'])
        valid_count = (3 <= n <= expected) if key == 'pointflow_inputs' else n == expected
        if not valid_count or not np.isfinite(inputs['anchor_xyz']).all():
            raise ValueError(f'{key}: invalid finite anchor count {n} (budget {expected})')
        metadata = dict(meta, timing=timing, start_frame=meta['frame_id'],
                        target_seconds=np.arange(33)/20., point_block_seconds=np.arange(4,33,4)/20.,
                        uv_to_video=np.array([[1,0,0],[0,1,0]],np.float32),
                        video_size_wh=np.array([640,720]), empty_anchor=False)
        sample = dict(inputs=inputs, metadata=metadata,
                      targets=dict(displacement=np.zeros((32,n,3),np.float32),
                                   valid=np.zeros((32,n),bool)))
        result.append(sample)
    # Same pixel-center resize as the training video transform. The head occupies
    # the first 480 rows of the 640x720 three-camera composite, before padding.
    h,w,rh,rw = map(int, image_size.reshape(-1).tolist())
    resize_pointflow_metadata(result[0], (640,720), (rw,rh), (w,h))
    return result


class PointFKPolicy(BundlePolicy):
    def __init__(self, config):
        environment = json.loads(Path(config['model_environment']).read_text())
        os.environ.update({k:str(v) for k,v in environment.items()})
        os.environ.update(REAL_ACTION_VARIANT='', REAL_ACTION_STATS_PATH='', REAL_ACTION_STATS_SHA256='',
                          FK_ROLLOUT='false', COSMOS_GPU_VIDEO_AUGMENTATION='false')
        self._geometry = None
        self._observation_lock = threading.RLock()
        self.prediction_root = Path(config.get('prediction_dir') or os.environ.get(
            'COSMOS_PREDICTION_DIR', str(Path(config['output_dir']) / 'predictions')))
        self._episode_index = 0
        self._query_index = 0
        super().__init__(config)
        from cosmos_framework.utils.bench2dex_contract import JOINT_NAMES
        if list(JOINT_NAMES) != self.contract['joint_names']:
            raise ValueError('Normalizer order differs from the Bench2Dex model joint order')
        net = self.backend.model.net
        if not hasattr(net, 'pointflow_branch') or not hasattr(net, 'fk_branch'):
            raise RuntimeError('Checkpoint model did not instantiate both PointFlow and FK branches')
        self.compile_transformer = bool(config.get('compile_transformer', False))
        if self.compile_transformer:
            from policy.Cosmos.compiled_forward import compile_forward
            self.compile_log = Path(config['output_dir']) / 'compile_calls.jsonl'
            self.compile_dynamic = bool(config.get('compile_dynamic', True))
            self.compile_mode = config.get('compile_mode', 'default')
            self.compiled_forward = compile_forward(
                net.language_model.forward, dynamic=self.compile_dynamic,
                mode=self.compile_mode,
                on_fallback=lambda error: print('[pointfk-compile-fallback] ' + json.dumps(error), flush=True))
            net.language_model.forward = self.compiled_forward
            print('[pointfk-compile] Transformer compilation enabled; preserve precision; dynamic=' + str(self.compile_dynamic), flush=True)
        generate = self.backend.model.generate_samples_from_batch

        def checked_generate(*args, **kwargs):
            output = generate(*args, **kwargs)
            n_point = len(self._geometry['pointflow_inputs']['point_ids'])
            for name, n in [('pointflow',n_point),('fk',42)]:
                value = output[name][0]
                if tuple(value.shape) != (32,n,3) or not bool(value.isfinite().all()):
                    raise RuntimeError(f'Invalid joint {name} predictions')
            self.last_geometry_predictions = {k:output[k][0].detach().float().cpu().numpy()
                                              for k in ('pointflow','fk')}
            return output
        self.backend.model.generate_samples_from_batch = checked_generate

    def augment_batch(self, batch):
        if self._geometry is None:
            raise ValueError('Point/FK policy requires synchronized live geometry')
        point, fk = geometry_samples(self._geometry, batch['image_size'])
        batch['pointflow'], batch['fk'] = [point], [fk]
        batch['sequence_plan'][0].has_point = True
        return align_training_prompt(batch, self.backend.input_video_key, self.backend.config.model.task)

    def get_action(self, observation):
        geometry = observation.get('current_geometry')
        if geometry is None:
            raise ValueError('Missing current_geometry; enable --live-pointfk in the simulator')
        if geometry['metadata']['frame_id'] != observation.get('geometry_frame_id'):
            raise ValueError('Geometry/RGB frame mismatch')
        meta = geometry['metadata']
        if 'qpos' in meta:
            order = [meta['joint_names'].index(n) for n in observation['joint_names']]
            np.testing.assert_allclose(np.asarray(meta['qpos'])[order],
                                       observation['joint_action']['vector'], rtol=0, atol=1e-6,
                                       err_msg='Geometry and action state are not synchronized')
        with self._observation_lock:
            self._geometry = geometry
            try:
                if self.compile_transformer:
                    from torch._dynamo.utils import counters
                    graphs_before = counters['stats']['unique_graphs']
                actions = super().get_action(observation)
                if self.compile_transformer:
                    row = dict(episode_index=self._episode_index, query_index=self._query_index,
                               unique_graphs=counters['stats']['unique_graphs'],
                               new_graphs=counters['stats']['unique_graphs'] - graphs_before,
                               dynamic=self.compile_dynamic, mode=self.compile_mode,
                               compiled_active=self.compiled_forward.failure is None,
                               fallback=self.compiled_forward.failure)
                    with self.compile_log.open('a') as stream:
                        stream.write(json.dumps(row) + '\n')
                    print('[pointfk-compile] ' + json.dumps(row), flush=True)
                from policy.Cosmos.prediction_recording import save_prediction
                folder = self.prediction_root / f'episode_{self._episode_index:06d}_seed_{self.backend.seed}'
                path = folder / f'query_{self._query_index:06d}.npz'
                seconds = save_prediction(path, geometry, self.last_geometry_predictions, actions,
                                          seed=self.backend.seed, query_index=self._query_index)
                print('[pointfk-prediction] ' + json.dumps(dict(path=str(path), save_seconds=seconds)), flush=True)
                self._query_index += 1
                return actions
            finally:
                self._geometry = None

    def reset(self, seed=None):
        with self._observation_lock:
            super().reset(seed)
            self._episode_index += 1
            self._query_index = 0
