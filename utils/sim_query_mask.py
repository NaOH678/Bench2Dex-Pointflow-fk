"""Offline simulator-geometry query masks. No future state is used."""
import numpy as np
from numba import njit


@njit(cache=True)
def rasterize(triangles_camera, triangle_labels, K, height, width):
    """Perspective-correct triangle z-buffer at integer pixel centers."""
    depth = np.full((height, width), np.inf, np.float32)
    labels = np.zeros((height, width), np.int32)
    for i in range(len(triangles_camera)):
        tri = triangles_camera[i]
        if np.min(tri[:, 2]) <= 1e-5:
            continue
        x = tri[:, 0]/tri[:, 2]*K[0, 0]+K[0, 2]
        y = tri[:, 1]/tri[:, 2]*K[1, 1]+K[1, 2]
        xmin, xmax = max(0, int(np.ceil(np.min(x)))), min(width-1, int(np.floor(np.max(x))))
        ymin, ymax = max(0, int(np.ceil(np.min(y)))), min(height-1, int(np.floor(np.max(y))))
        denom = (y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2])
        if abs(denom) < 1e-12:
            continue
        for v in range(ymin, ymax+1):
            for u in range(xmin, xmax+1):
                a = ((y[1]-y[2])*(u-x[2])+(x[2]-x[1])*(v-y[2]))/denom
                b = ((y[2]-y[0])*(u-x[2])+(x[0]-x[2])*(v-y[2]))/denom
                c = 1-a-b
                if min(a, b, c) < -1e-6:
                    continue
                z = 1/(a/tri[0, 2]+b/tri[1, 2]+c/tri[2, 2])
                if z < depth[v, u]:
                    depth[v, u] = z
                    labels[v, u] = triangle_labels[i]
    return depth, labels


class SimMaskRenderer:
    """Cache episode visual meshes; render each window's current-state mask."""
    def __init__(self, source, assets, scene, camera='cam_overhead', *, fast=False):
        import json
        from pathlib import Path
        import yaml
        from pxr import Usd
        from scipy.spatial.transform import Rotation
        from utils.sim_surface_gt import KinematicTree, robot_visuals, object_visuals, transform
        self.source, self.camera = source, camera
        self.fast = fast
        assets = Path(assets)
        from utils.sim_robot_profile import profile
        robot = profile(source, assets)
        self.tree = robot['tree']
        self.meshes = robot_visuals(Usd.Stage.Open(str(robot['usd'])))
        for mesh in self.meshes:
            mesh['group'] = 'hand' if mesh['name'] in robot['hand_links'] else 'arm'
        self.base = robot['base']
        self.names = [n.decode() for n in source['robot/joint_names'][:]]
        cfg = yaml.safe_load(Path(scene).read_text())
        sample = json.loads(source['meta/scene_generalization_sample'][()])
        clutter = {x['obj_id']: x for x in sample.get('clutter', {}).get('tabletop', {}).get('assets', []) if x.get('obj_id')}
        keys = json.loads(source['meta/object_asset_keys'][()])
        self.object_names = set(source['objects'])
        for name, original in json.loads(source['meta/object_asset_paths'][()]).items():
            if name not in self.object_names:
                continue
            path = assets/original.split('dex2bench_dataset/', 1)[1]
            scale = cfg['assets'][keys[name]]['scale'] if keys[name] in cfg['assets'] else clutter[name]['scale']
            self.meshes.extend(object_visuals(Usd.Stage.Open(str(path)), scale, name))
        self.K = source[f'cameras/{camera}/intrinsic'][:]

    def render(self, row, grid_step=4, erosion=1):
        import cv2
        from utils.sim_surface_gt import pose_matrix
        from utils.rgbd_pointflow import image_queries, TrackingConfig
        f = self.source
        cam = f[f'cameras/{self.camera}']
        depth = cam['depth'][row]
        poses = {name: pose_matrix(f[f'objects/{name}/pose_world'][row]) for name in self.object_names}
        mask = self.render_current(depth=depth, intrinsic=self.K,
            world_from_camera=cam['extrinsic_world_from_cam'][row],
            joint_names=self.names, qpos=f['robot/qpos'][row],
            world_from_robot=self.base, object_poses=poses, erosion=erosion)
        rgb = cv2.imdecode(cam['rgb'][row], cv2.IMREAD_COLOR)
        queries = image_queries(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), TrackingConfig(grid_step=grid_step, border=grid_step//2, texture_filter=False))
        xy = np.rint(queries).astype(int)
        region = mask[xy[:, 1], xy[:, 0]]
        keep = region > 0
        return queries[keep], region[keep], mask, rgb, depth

    @classmethod
    def from_runtime(cls, runtime, robot_asset):
        """Cache the live scene's actual meshes/scales; never read recorded poses."""
        from pathlib import Path
        from pxr import Usd
        from utils.sim_surface_gt import KinematicTree, robot_visuals, object_visuals
        obj = cls.__new__(cls)
        obj.fast = True
        robot_asset = Path(robot_asset)
        obj.tree = KinematicTree.from_urdf(robot_asset/'urdf/Multi_UR5_wuji_with_flange.urdf')
        obj.meshes = robot_visuals(Usd.Stage.Open(str(robot_asset/'usd/Multi_UR5_wuji_with_flange.usd')))
        for name,path in runtime['object_asset_paths'].items():
            body = runtime['interactive_objects'].get(name)
            if body is None:
                continue
            scale = body.cfg.spawn.scale
            obj.meshes.extend(object_visuals(Usd.Stage.Open(str(path)), scale, name))
        obj.object_names = {m['name'] for m in obj.meshes if m['group']=='object'}
        return obj

    def render_current(self, *, depth, intrinsic, world_from_camera, joint_names,
                       qpos, world_from_robot, object_poses, erosion=1):
        import cv2
        from utils.sim_surface_gt import apply
        from utils.rgbd_pointflow import isaac_world_from_optical
        optical = np.linalg.inv(isaac_world_from_optical(world_from_camera))
        fk = self.tree.fk(dict(zip(joint_names, qpos)))
        triangles, labels = [], []
        for m in self.meshes:
            pose = object_poses[m['name']] if m['group'] == 'object' else world_from_robot @ fk[m['name']]
            if self.fast:
                T = optical @ pose
                tri = (m['triangles'].reshape(-1,3) @ T[:3,:3].T + T[:3,3]).reshape(-1,3,3)
            else:
                tri = apply(optical @ pose, m['triangles'])
            triangles.append(tri)
            labels.append(np.full(len(tri), {'hand': 2, 'object': 3, 'arm': 0}[m['group']], np.int32))
        if self.fast:
            from utils.sim_mask_fast import rasterize_scalar
            render_triangles = rasterize_scalar
        else:
            render_triangles = rasterize
        z, mask = render_triangles(np.concatenate(triangles), np.concatenate(labels), intrinsic, *depth.shape)
        mask[(~np.isfinite(depth)) | (depth <= 0) | (np.abs(z-depth) > .005)] = 0
        if erosion:
            kernel = np.ones((2*erosion+1, 2*erosion+1), np.uint8)
            mask = sum(cv2.erode((mask == label).astype(np.uint8), kernel)*label for label in [2, 3]).astype(np.int32)
        return mask


def complete_window_starts(sim_steps, physics_dt, homing_start, fps=20, steps=32, stride=32, action_valid=None):
    """Only uniform, non-homing windows; add a final overlapping tail window."""
    times = np.asarray(sim_steps)*physics_dt
    end = int(np.searchsorted(sim_steps, homing_start))
    candidates = [s for s in range(max(0, end-steps))
                  if np.allclose(np.diff(times[s:s+steps+1]), 1/fps, atol=1e-8, rtol=0)
                  and (action_valid is None or np.all(action_valid[s:s+steps]))]
    if not candidates:
        return [], end
    starts = list(range(candidates[0], candidates[-1]+1, stride))
    starts = [s for s in starts if s in candidates]
    if starts[-1] != candidates[-1]:
        starts.append(candidates[-1])
    return starts, end
