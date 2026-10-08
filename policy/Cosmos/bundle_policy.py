"""Use the supplied training source's Bench2Dex batch construction and loader."""
import json
import os
from pathlib import Path
import sys
import time
import typing

from policy.Cosmos.contract import load_contract, prepare_observation


def normalize_bundle_batch(batch, normalization):
    """Normalize observed state; let Cosmos invert generated actions exactly once."""
    if normalization['kind'] == 'none':
        return batch
    import torch
    from cosmos_framework.data.generator.action.action_processing import (
        ActionAffineNormalization, ActionProcessingRecord, make_batched_action_processing_fields,
    )
    if normalization['kind'] != 'affine':
        raise ValueError('Unsupported bundle action normalization')
    normalizer = ActionAffineNormalization(
        offset=torch.tensor(normalization['offset'], dtype=torch.float32),
        scale=torch.tensor(normalization['scale'], dtype=torch.float32),
        forward_clamp=normalization.get('forward_clamp'))
    action = batch['action'][0][0]
    action[0, :52] = normalizer.normalize_action(action[0, :52])
    batch.update(make_batched_action_processing_fields(
        ActionProcessingRecord(raw_action_dim=52, action_normalizer=normalizer), batch_size=1))
    return batch


class BundlePolicy:
    def __init__(self, config):
        if config.get('fixture_only'):
            raise ValueError('Cannot run an input fixture as a trained policy')
        self.contract = load_contract(config['contract_path'])
        os.environ['COSMOS_TRAINING'] = '0'
        os.environ['COSMOS_FLASH2_VARLEN'] = '1'
        # Python 3.11 lacks the typing-only decorator used by the source.
        if not hasattr(typing, 'override'):
            from typing_extensions import override
            typing.override = override
        sys.path.insert(0, str(Path(config['worldact_root']).resolve()))
        from cosmos_framework.inference.common.init import init_script
        init_script(training=False)
        from cosmos_framework.inference.robot_policy.bench2dex import Bench2DexPolicy
        options = json.loads(Path(config['bundle_deployment']).read_text())
        options.update(checkpoint_path=config['checkpoint_path'],
                       config_file=config['training_config_path'],
                       execution_horizon=int(config.get('execute_steps', 4)),
                       num_steps=int(config.get('num_steps', 4)),
                       guidance=float(config.get('guidance', 3)),
                       output_dir=config['output_dir'])
        normalization = self.contract['normalization']
        owner = self
        self.preprocess_observed_frame_only = bool(config.get("preprocess_observed_frame_only", True))
        class ContractBench2DexPolicy(Bench2DexPolicy):
            def _build_batch(self, images_rgb, right_state):
                if owner.preprocess_observed_frame_only:
                    from policy.Cosmos.observed_frame_batch import build_observed_frame_batch
                    batch = build_observed_frame_batch(self, images_rgb, right_state)
                else:
                    batch = super()._build_batch(images_rgb, right_state)
                batch = normalize_bundle_batch(batch, normalization)
                return owner.augment_batch(batch)
        self.backend = ContractBench2DexPolicy(options)
        self.profile_path = Path(config['output_dir']) / 'local_inference.jsonl'
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)

    def augment_batch(self, batch):
        return batch

    def reset(self, seed=None):
        with self.backend._lock:
            if seed is not None:
                self.backend.seed = int(seed)
            self.backend.reset_model()

    def get_action(self, observation):
        import torch
        prepare_observation(observation, self.contract)
        observation = dict(observation)
        observation['joint_action'] = dict(observation['joint_action'],
                                          qpos=observation['joint_action']['vector'])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        actions = self.backend.get_action(observation)
        torch.cuda.synchronize()
        profile = dict(seconds=time.perf_counter()-started, returned_actions=len(actions),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       cuda_free_bytes=torch.cuda.mem_get_info()[0])
        with self.profile_path.open('a') as stream:
            stream.write(json.dumps(profile)+'\n')
        print('[cosmos-local] '+json.dumps(profile), flush=True)
        return actions
