"""Compare optimized complete batches to the installed training-source adapter."""
from dataclasses import fields,is_dataclass
from types import SimpleNamespace
import typing
import numpy as np
import pytest
if not hasattr(typing,'override'):
    from typing_extensions import override
    typing.override=override
pytest.importorskip('cosmos_framework.inference.robot_policy.bench2dex')
import torch
from cosmos_framework.inference.robot_policy.bench2dex import Bench2DexPolicy
from policy.Cosmos.observed_frame_batch import build_observed_frame_batch


def assert_equal(a,b):
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):assert_equal(x,y)
    elif is_dataclass(a):
        for f in fields(a):assert_equal(getattr(a,f.name),getattr(b,f.name))
    else:assert a==b


def adapter():
    return SimpleNamespace(view_width=640,device='cpu',input_video_key='video',
        config=SimpleNamespace(model=SimpleNamespace(native_chunk_size=32,max_action_dim=64,
            native_action_dim=52,resolution='480',task='Put the condiments into the box',domain_name='bench2dex_wuji'),
            deployment=SimpleNamespace(action_rate_hz=20.)))


def compare(images,q):
    obj=adapter();old=Bench2DexPolicy._build_batch(obj,images,q);new=build_observed_frame_batch(obj,images,q)
    assert_equal(old,new)
    assert new['video'][0][0].shape==(3,33,640,640)
    assert torch.count_nonzero(new['video'][0][0][:,1:])==0


@pytest.mark.parametrize('frame',[0,250,500,999])
def test_recorded_observations_exactly_equal(frame):
    from pathlib import Path
    import cv2,h5py
    p=Path('outputs/cosmos_local/pointfk_3000/live_episode_exec32_same_seed/episodes/failure/episode_000001.hdf5')
    if not p.exists():pytest.skip('Local closed-loop recording unavailable')
    with h5py.File(p) as f:
        images={c:cv2.cvtColor(cv2.imdecode(f[f'cameras/{c}/rgb'][frame],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB) for c in ['cam_overhead','cam_wrist_left','cam_wrist_right']}
        compare(images,f['robot/qpos'][frame].astype(np.float32))


@pytest.mark.parametrize('fill',[0,255,'random'])
def test_full_batch_and_zero_future_equivalence(fill):
    rng=np.random.default_rng(17)
    images={c:(rng.integers(0,256,(480,640,3),dtype=np.uint8) if fill=='random' else np.full((480,640,3),fill,np.uint8)) for c in ['cam_overhead','cam_wrist_left','cam_wrist_right']}
    compare(images,rng.normal(size=52).astype(np.float32))
