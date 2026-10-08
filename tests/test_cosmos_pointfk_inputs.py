"""Conditioning contract checks; run with the matching WorldAct on PYTHONPATH."""
import typing
import numpy as np
import pytest

if not hasattr(typing, 'override'):
    from typing_extensions import override
    typing.override = override
pytest.importorskip('cosmos_framework.data.pointflow_window')
from policy.Cosmos.pointfk_policy import geometry_samples
from policy.Cosmos.anchor_selection import hand_guided_fps


def test_inference_prompt_matches_cached_training_transform():
    import json
    import torch
    from policy.Cosmos.pointfk_policy import align_training_prompt
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

    task = 'Put the condiments into the box'
    # Training stores only the anchor RGB with a 33-frame cached video extent.
    training = ActionTransformPipeline(tokenizer_config=None, cfg_dropout_rate=0,
                                      max_action_dim=64, format_prompt_as_json=True)(dict(
        video=torch.zeros((3, 1, 720, 640), dtype=torch.uint8),
        action=torch.zeros((33, 52)), ai_caption=task,
        conditioning_fps=torch.tensor(20), mode='wam', viewpoint='concat_view',
        additional_view_description='Head on top; left wrist bottom-left; right wrist bottom-right.',
        cached_video_num_frames=33), '480')
    batch = dict(video=[[torch.zeros((3, 33, 1, 1))]], action=[[torch.zeros((33, 64))]],
                 conditioning_fps=[torch.tensor(20)], image_size=training['image_size'][None],
                 ai_caption=['old baseline prompt'], prompt=['old baseline prompt'])
    align_training_prompt(batch, 'video', task)
    assert json.loads(batch['prompt'][0]) == training['ai_caption']
    assert batch['prompt'] == batch['ai_caption']
    assert json.loads(batch['prompt'][0])['duration'] == '1s'


def current_geometry():
    return dict(metadata=dict(frame_id=4, uses_future_frames=False,
                              coordinate='fixed_optical_camera', units='metre'),
        pointflow_inputs=dict(point_ids=np.arange(1024),anchor_xyz=np.ones((1024,3),np.float32)),
        fk_inputs=dict(point_ids=np.arange(42),anchor_xyz=np.ones((42,3),np.float32)))


def test_current_only_payload_and_pixel_center_alignment():
    import torch
    g=current_geometry()
    # Deliberately irrelevant future keys must not become conditioning.
    g['future_xyz']=np.full((32,1024,3),999.)
    point,fk=geometry_samples(g,torch.tensor([[640,640,640,569]]))
    for sample,n in [(point,1024),(fk,42)]:
        assert sample['targets']['displacement'].shape==(32,n,3)
        assert not sample['targets']['displacement'].any()
        assert not sample['targets']['valid'].any()
        assert 'future_xyz' not in sample['inputs']
    affine=point['metadata']['uv_to_video']
    np.testing.assert_allclose(affine,[[569/640,0,(569/640-1)/2],
                                     [0,640/720,(640/720-1)/2]],atol=1e-7)
    np.testing.assert_array_equal(point['metadata']['video_size_wh'],[640,640])
    point['inputs']['anchor_xyz'][0]=2
    assert (g['pointflow_inputs']['anchor_xyz']==1).all()


def test_future_or_wrong_coordinate_rejected():
    import torch
    g=current_geometry();g['metadata']['uses_future_frames']=True
    with pytest.raises(ValueError,match='current camera-frame'):
        geometry_samples(g,torch.tensor([640,640,640,569]))


def test_selection_matches_training_including_local_deficit():
    from cosmos_framework.data.pointflow_anchor_selection import hand_guided_fps as training
    rng=np.random.default_rng(42)
    xyz=rng.normal(size=(1500,3)).astype(np.float32)*.1
    for fk in [rng.normal(size=(42,3))*.1,np.ones((42,3))*9]:
        got,detail=hand_guided_fps(xyz,fk,1024,.05)
        expected,expected_detail=training(xyz,fk,1024,.05)
        np.testing.assert_array_equal(got,expected)
        assert detail==expected_detail and len(np.unique(got))==1024


def test_reject_stale_joint_snapshot_before_model_call():
    from policy.Cosmos.pointfk_policy import PointFKPolicy
    policy=PointFKPolicy.__new__(PointFKPolicy)
    geom=current_geometry()
    geom['metadata'].update(joint_names=['a','b'],qpos=[1.,2.])
    obs=dict(current_geometry=geom,geometry_frame_id=4,joint_names=['b','a'],
             joint_action=dict(vector=np.array([2.,9.])))
    with pytest.raises(AssertionError,match='not synchronized'):
        policy.get_action(obs)
    obs['geometry_frame_id']=5
    with pytest.raises(ValueError,match='frame mismatch'):
        policy.get_action(obs)


def test_partial_visibility_keeps_ragged_point_count():
    import torch
    g=current_geometry()
    g['pointflow_inputs']={k:v[:500] for k,v in g['pointflow_inputs'].items()}
    point,_=geometry_samples(g,torch.tensor([640,640,640,569]))
    assert point['targets']['displacement'].shape==(32,500,3)
