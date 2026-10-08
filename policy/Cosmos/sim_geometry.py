"""Live Isaac snapshots -> current Point/FK RPC payload, once per action query."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from policy.Cosmos.anchor_selection import hand_guided_fps
from policy.Cosmos.live_geometry import CurrentFrameGeometry
from utils.sim_query_mask import SimMaskRenderer
from utils.sim_surface_gt import pose_matrix


class SimGeometryProvider:
    def __init__(self, runtime, robot_asset, log_dir):
        self.renderer = SimMaskRenderer.from_runtime(runtime, robot_asset)
        self.geometry = CurrentFrameGeometry(
            Path(robot_asset)/'urdf/Multi_UR5_wuji_with_flange.urdf',
            selector=lambda xyz,fk: hand_guided_fps(xyz,fk,1024,.05))
        self.log_path = Path(log_dir)/'live_geometry.jsonl'
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self._saved_snapshot = False

    def compute(self, frame, robot_art, object_states, *, frame_id, sim_time_sec):
        if frame is None or frame.depth_m is None or frame.rgb is None:
            raise ValueError('Live Point/FK requires synchronized overhead RGB and optical depth')
        if frame.rgb.shape != (480,640,3):
            raise ValueError(f'Expected training overhead resolution 640x480, got {frame.rgb.shape}')
        start = perf_counter()
        qpos = robot_art.data.joint_pos[0].detach().cpu().numpy().copy()
        names = list(robot_art.joint_names)
        root = robot_art.data.root_state_w[0,:7].detach().cpu().numpy()
        # Isaac root quaternion is wxyz; pose_matrix takes xyzw.
        base = pose_matrix(np.r_[root[:3],root[4:7],root[3]])
        poses = {n:pose_matrix(object_states[n]['pose_world']) for n in self.renderer.object_names}
        params = dict(intrinsic=frame.intrinsic,world_from_camera=frame.extrinsic_world_from_cam,
                      joint_names=names,qpos=qpos,world_from_robot=base)
        mask = self.renderer.render_current(depth=frame.depth_m,object_poses=poses,**params)
        mask_sec = perf_counter()-start
        result = self.geometry.compute(rgb=frame.rgb,depth_m=frame.depth_m,visible_mask=mask,
            frame_id=frame_id,sim_time_sec=sim_time_sec,**params)
        result['timings'].update(mask_sec=mask_sec, total_with_mask_sec=perf_counter()-start)
        result['metadata'].update(joint_names=names, qpos=qpos.tolist())
        if not self._saved_snapshot:
            np.savez_compressed(self.log_path.parent/'first_live_geometry.npz',
                rgb=frame.rgb, mask=mask, depth=frame.depth_m, intrinsic=frame.intrinsic,
                point_xyz=result['pointflow_inputs']['anchor_xyz'],
                point_uv=result['pointflow_inputs']['anchor_uv'],
                fk_xyz=result['fk_inputs']['anchor_xyz'], fk_uv=result['fk_inputs']['anchor_uv'])
            self._saved_snapshot = True
        with self.log_path.open('a') as stream:
            stream.write(json.dumps(dict(frame_id=frame_id,**result['timings'],
                points=len(result['pointflow_inputs']['point_ids']),mask_pixels=int((mask>0).sum())))+'\n')
        return result
