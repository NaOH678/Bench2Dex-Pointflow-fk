import numpy as np
import pytest
from utils.sim_query_mask import rasterize
from utils.sim_mask_fast import rasterize_scalar


@pytest.mark.parametrize('render', [rasterize, rasterize_scalar])
def test_zbuffer_keeps_nearer_hand_and_unlabeled_occluder(render):
    back = np.array([[-2., -2, 2], [2, -2, 2], [0, 2, 2]])
    front = back*.5
    K = np.array([[2., 0, 2], [0, 2, 2], [0, 0, 1]])
    depth, mask = render(np.array([back, front]), np.array([3, 2]), K, 5, 5)
    assert mask[2, 2] == 2
    assert depth[2, 2] == 1
    _, mask = render(np.array([back, front]), np.array([3, 0]), K, 5, 5)
    assert mask[2, 2] == 0


@pytest.mark.parametrize('render', [rasterize, rasterize_scalar])
def test_perspective_depth_uses_reciprocal_interpolation(render):
    # Projected vertices (0,0), (4,0), (0,4); center (1,1) weights .5,.25,.25.
    tri = np.array([[[0., 0, 1], [8, 0, 2], [0, 16, 4]]])
    depth, mask = render(tri, np.array([2]), np.eye(3), 5, 5)
    assert mask[1, 1] == 2
    np.testing.assert_allclose(depth[1, 1], 1/(.5+.25/2+.25/4))


def test_fast_raster_preserves_clipping_degeneracy_and_depth_order():
    rng=np.random.default_rng(19)
    triangles=rng.normal(size=(300,3,3));triangles[:,:,2]+=1
    triangles[0]=0  # degenerate and on camera plane
    triangles[1,:,2]=-1  # behind camera
    triangles[2]=triangles[3]  # identical surfaces preserve reference tie behavior
    labels=rng.choice([0,2,3],len(triangles)).astype(np.int32)
    K=np.array([[30.,0,32],[0,30,24],[0,0,1]])
    old=rasterize(triangles,labels,K,48,64)
    new=rasterize_scalar(triangles,labels,K,48,64)
    for a,b in zip(old,new):np.testing.assert_array_equal(a,b)


def test_windows_exclude_irregular_interval_and_homing():
    from utils.sim_query_mask import complete_window_starts
    steps = np.r_[0, 2+3*np.arange(79)]
    starts, end = complete_window_starts(steps, 1/60, steps[72], stride=32)
    assert end == 72
    assert starts == [1, 33, 39]
    assert all(s+32 < end for s in starts)


def test_uniform_first_interval_still_excludes_invalid_initial_command():
    from utils.sim_query_mask import complete_window_starts
    steps = 3*np.arange(80)
    valid = np.ones(80, bool)
    valid[0] = False
    starts, end = complete_window_starts(steps, 1/60, steps[72], action_valid=valid)
    assert starts == [1, 33, 39]


def test_live_mask_uses_current_object_pose_and_observed_occlusion():
    from types import SimpleNamespace
    from utils.sim_query_mask import SimMaskRenderer
    from utils.rgbd_pointflow import isaac_world_from_optical
    renderer=SimMaskRenderer.__new__(SimMaskRenderer)
    renderer.fast=True
    renderer.tree=SimpleNamespace(fk=lambda values: {})
    renderer.meshes=[dict(name='bottle',group='object',triangles=np.array([
        [[-2.,-2,2],[2,-2,2],[0,2,2]]]))]
    K=np.array([[2.,0,2],[0,2,2],[0,0,1]])
    args=dict(depth=np.full((5,5),2,np.float32),intrinsic=K,
              world_from_camera=np.linalg.inv(isaac_world_from_optical(np.eye(4))),joint_names=[],qpos=[],
              world_from_robot=np.eye(4),erosion=0)
    mask=renderer.render_current(**args,object_poses={'bottle':np.eye(4)})
    assert mask[2,2]==3
    moved=np.eye(4);moved[0,3]=10
    assert not renderer.render_current(**args,object_poses={'bottle':moved}).any()
    args['depth'][2,2]=1  # unmodeled closer surface occludes this pixel
    assert renderer.render_current(**args,object_poses={'bottle':np.eye(4)})[2,2]==0
