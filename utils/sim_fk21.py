"""Explicit URDF-native 21-point convention for each simulated Wuji hand.

This defines this robot's landmarks; it does not claim bitwise compatibility
with an external human-hand annotation generator. All joint angles participate
in FK, including the basal joint whose origin is not a separate landmark.
"""
import numpy as np

SIDES = ('left', 'right')
KEYPOINT_NAMES = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip']
for finger in ('index', 'middle', 'ring', 'pinky'):
    KEYPOINT_NAMES += [f'{finger}_{joint}' for joint in ('mcp', 'pip', 'dip', 'tip')]


def hand_links(side):
    if side not in SIDES:
        raise ValueError(side)
    links = [f'{side}_palm_link']
    for finger in range(1, 6):
        links += [f'{side}_finger{finger}_link{joint}' for joint in (2, 3, 4)]
        links.append(f'{side}_finger{finger}_tip_link')
    return links


def mapping_document():
    return dict(schema='bench2dex_wuji_urdf_fk21_v1', sides=list(SIDES),
        units='metre', convention='palm frame origin, then link2/link3/link4/tip origins for each finger',
        note='Robot landmarks using standard 21-point ordering; anatomical names label robot proxies.',
        mapping={side: [dict(index=i, name=name, link=link, local_xyz=[0., 0., 0.])
            for i, (name, link) in enumerate(zip(KEYPOINT_NAMES, hand_links(side)))] for side in SIDES})


def positions_from_transforms(transforms):
    return np.array([[transforms[link][:3, 3] for link in hand_links(side)] for side in SIDES])


def fk_window_sample(archive, frame_ids, episode, fps=20):
    from cosmos_framework.data.fk_window import FKTiming
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    rows = np.searchsorted(archive['source_frame_ids'], frame_ids)
    np.testing.assert_array_equal(archive['source_frame_ids'][rows], frame_ids)
    timing = FKTiming(fps=fps, steps=len(frame_ids)-1, steps_per_token=4)
    positions = archive['positions_camera'][rows].reshape(len(rows), 42, 3).astype(np.float32)
    assert np.isfinite(positions).all()
    anchor = positions[0]
    K = archive['intrinsics']
    projected = anchor @ K.T
    uv = projected[:, :2]/projected[:, 2:]
    width, height = archive['image_size_wh']
    return dict(inputs=dict(anchor_xyz=anchor.copy(), point_ids=np.arange(42, dtype=np.int64),
                            anchor_uv=uv.astype(np.float32), image_size_wh=np.array([width, height])),
        targets=dict(displacement=positions[1:]-anchor[None], valid=np.ones((timing.steps, 42), bool)),
        metadata=dict(episode=episode, start_frame=int(frame_ids[0]), raw_frame_ids=frame_ids.copy(), timing=timing,
            hand='both', hand_order=list(SIDES), coordinate='fixed_optical_camera',
            point_block_seconds=np.arange(4, timing.steps+1, 4)/fps,
            geometry_source='observed qpos + URDF; future qpos only supplies target labels',
            validity_semantics='FK geometry known from proprioception, independent of camera visibility',
            keypoint_names=[f'{side}_{name}' for side in SIDES for name in KEYPOINT_NAMES]))
