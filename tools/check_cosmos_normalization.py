"""Check real state normalization and Cosmos output inversion without loading weights."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import typing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--hdf5', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, cfg['worldact_root'])
    if not hasattr(typing, 'override'):
        from typing_extensions import override
        typing.override = override
    import cv2
    import h5py
    import numpy as np
    import torch
    from policy.Cosmos.contract import load_contract
    from policy.Cosmos.bundle_policy import normalize_bundle_batch
    from cosmos_framework.inference.robot_policy.bench2dex import Bench2DexPolicy
    from cosmos_framework.utils.bench2dex_contract import CAMERAS, JOINT_NAMES
    from cosmos_framework.data.generator.action.action_processing import ActionProcessor, get_action_processing_records

    contract = load_contract(cfg['contract_path'])
    assert contract['joint_names'] == list(JOINT_NAMES)
    norm = contract['normalization']
    assert norm['kind'] == 'affine' and norm.get('forward_clamp') is None
    policy = object.__new__(Bench2DexPolicy)
    policy.device, policy.view_width, policy.input_video_key = 'cpu', 640, 'video'
    policy.config = SimpleNamespace(model=SimpleNamespace(native_chunk_size=32, max_action_dim=64,
        native_action_dim=52, task='Normalization contract check', resolution='480', domain_name='bench2dex_wuji'),
        deployment=SimpleNamespace(action_rate_hz=20.))
    with h5py.File(a.hdf5) as source:
        names = [v.decode() for v in source['robot/joint_names'][:]]
        state = source['robot/qpos'][1][[names.index(n) for n in JOINT_NAMES]]
        images = {c: cv2.cvtColor(cv2.imdecode(source[f'cameras/{c}/rgb'][1], 1), cv2.COLOR_BGR2RGB) for c in CAMERAS}
    batch = normalize_bundle_batch(policy._build_batch(images, state), norm)
    action = batch['action'][0][0]
    offset, scale = np.asarray(norm['offset'], np.float32), np.asarray(norm['scale'], np.float32)
    np.testing.assert_array_equal(action[0, :52].numpy(), (state-offset)/scale)
    assert torch.count_nonzero(action[1:]) == 0 and torch.count_nonzero(action[:, 52:]) == 0
    record = get_action_processing_records(batch)[0]
    recovered = ActionProcessor.postprocess_action(action, record)[0].numpy()
    np.testing.assert_allclose(recovered, state, atol=2e-7)
    prediction = torch.linspace(-3, 3, 52)[None]
    raw = ActionProcessor.postprocess_action(torch.nn.functional.pad(prediction, (0, 12)), record)
    np.testing.assert_allclose(raw.numpy(), prediction.numpy()*scale+offset, atol=1e-7)
    result = dict(status='passed', observed_state_normalization_exact=True,
        future_inputs_and_padding_zero=True, prediction_denormalization_passed=True,
        state_roundtrip_max_rad=float(abs(recovered-state).max()), model_executed=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
