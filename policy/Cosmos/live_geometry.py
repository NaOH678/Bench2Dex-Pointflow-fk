"""Current-frame PointFlow/FK conditioning; never reads future frames or labels."""
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from utils.rgbd_pointflow import lift_depth, isaac_world_from_optical
from utils.sim_surface_gt import KinematicTree
from utils.sim_fk21 import hand_links, SIDES


def point_anchor(xyz, uv, rgb, intrinsic, *, max_points=16384, voxel_size=.02, seed=0):
    """Match WorldAct anchor normal estimation and voxel features on observed points."""
    import cv2
    xyz=np.asarray(xyz,np.float32);uv=np.asarray(uv,np.float32);K=np.asarray(intrinsic,np.float32)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or uv.shape!=(len(xyz),2):
        raise ValueError('Expected XYZ [N,3] and UV [N,2]')
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
        raise ValueError('Expected RGB uint8 [H,W,3]')
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError('Invalid optical camera intrinsics')
    if max_points<3 or not np.isfinite(voxel_size) or voxel_size<=0:
        raise ValueError('Invalid point budget or voxel size')
    h,w=rgb.shape[:2]
    good=np.isfinite(xyz).all(1)&(xyz[:,2]>0)&np.isfinite(uv).all(1)
    good&=(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
    candidates=np.flatnonzero(good)
    if len(candidates)<3:raise ValueError('Fewer than three observed surface points')
    ids=np.sort(np.random.default_rng(seed).choice(candidates,min(max_points,len(candidates)),replace=False))
    tree=cKDTree(xyz[candidates]);_,neighbors=tree.query(xyz[ids],k=min(16,len(candidates)),workers=1)
    local=xyz[candidates][neighbors].astype(np.float64);local-=local.mean(1,keepdims=True)
    values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',local,local)/local.shape[1])
    reliable=values[:,1]>1e-12;ids=ids[reliable];normal=vectors[reliable,:,0].astype(np.float32)
    if len(ids)<3:raise ValueError('Degenerate observed surface geometry')
    normal[np.sum(normal*xyz[ids],axis=1)>0]*=-1
    xyz,uv=xyz[ids],uv[ids]
    color=cv2.remap(rgb,uv[:,0,None],uv[:,1,None],cv2.INTER_LINEAR).reshape(-1,3)
    shift=(xyz.min(0)+xyz.max(0))/2;shift[2]=xyz[:,2].min()
    centered=xyz-shift;grid=np.floor(centered/voxel_size).astype(np.int64);grid-=grid.min(0)
    _,representatives,inverse=np.unique(grid,axis=0,return_index=True,return_inverse=True)
    features=np.concatenate([centered,color.astype(np.float32)/255,normal],axis=1)
    return dict(point_ids=ids,anchor_xyz=xyz,anchor_uv=uv,normal=normal,color=color,
        coord=centered[representatives],feat=features[representatives],grid_coord=grid[representatives],
        original_to_voxel=inverse,voxel_representatives=representatives,coord_shift=shift,
        intrinsics_normalized=(np.diag([1/w,1/h,1])@K).astype(np.float32),image_size_wh=np.array([w,h],np.int64))


class CurrentFrameGeometry:
    """Cache the URDF tree; evaluate exactly one synchronized simulator observation."""
    def __init__(self, urdf, *, grid_step=2,max_points=16384,voxel_size=.02,selector=None):
        if type(grid_step) is not int or grid_step<1:raise ValueError('Invalid grid step')
        self.tree=KinematicTree.from_urdf(urdf)
        self.grid_step,self.max_points,self.voxel_size=grid_step,max_points,voxel_size
        self.selector=selector

    def compute(self, *, rgb,depth_m,visible_mask,intrinsic,world_from_camera,
                joint_names,qpos,world_from_robot,frame_id,sim_time_sec,seed=0):
        start=perf_counter()
        if depth_m.shape!=rgb.shape[:2] or visible_mask.shape!=depth_m.shape:
            raise ValueError('RGB, optical depth and visible mask must share one image grid')
        qpos=np.asarray(qpos)
        if len(set(joint_names))!=len(joint_names) or qpos.shape!=(len(joint_names),) or not np.isfinite(qpos).all():
            raise ValueError('Invalid current joint state')
        yy,xx=np.mgrid[0:depth_m.shape[0]:self.grid_step,0:depth_m.shape[1]:self.grid_step]
        valid=np.isin(visible_mask[yy,xx],[2,3])&np.isfinite(depth_m[yy,xx])&(depth_m[yy,xx]>0)
        uv=np.c_[xx[valid],yy[valid]].astype(np.float32)
        _,valid,xyz=lift_depth(uv,depth_m,intrinsic)
        point=point_anchor(xyz[valid],uv[valid],rgb,intrinsic,max_points=self.max_points,
                           voxel_size=self.voxel_size,seed=seed)
        point_sec=perf_counter()-start;fk_start=perf_counter()
        fk=self.tree.fk(dict(zip(joint_names,qpos)))
        optical_from_robot=np.linalg.inv(isaac_world_from_optical(world_from_camera))@world_from_robot
        positions=np.asarray([(optical_from_robot@fk[link])[:3,3] for side in SIDES for link in hand_links(side)],np.float32)
        if not np.isfinite(positions).all() or np.any(positions[:,2]<=0):
            raise ValueError('FK points behind camera or nonfinite; check calibration')
        projected=positions@np.asarray(intrinsic).T
        fk_input=dict(anchor_xyz=positions,point_ids=np.arange(42,dtype=np.int64),
                      anchor_uv=(projected[:,:2]/projected[:,2:]).astype(np.float32),image_size_wh=point['image_size_wh'].copy())
        fk_sec=perf_counter()-fk_start
        selection_start=perf_counter();selection_detail=None
        if self.selector is not None:
            keep,selection_detail=self.selector(point['anchor_xyz'],positions)
            point=subset_anchor(point,keep,self.voxel_size)
        selection_sec=perf_counter()-selection_start
        return dict(pointflow_inputs=point,fk_inputs=fk_input,
            metadata=dict(frame_id=int(frame_id),sim_time_sec=float(sim_time_sec),coordinate='fixed_optical_camera',
                          units='metre',hand_order=list(SIDES),uses_future_frames=False,uses_gt_query_mask=True,
                          point_selection='current visible mask grid; no trajectory-lifetime filtering',
                          selection_detail=selection_detail),
            timings=dict(point_anchor_sec=point_sec,fk_sec=fk_sec,selection_sec=selection_sec,total_sec=perf_counter()-start))


def subset_anchor(point, keep, voxel_size=.02):
    """Preserve selected normals/colors, then re-voxelize as in training preprocessing."""
    keep=np.asarray(keep)
    if keep.ndim!=1 or not np.issubdtype(keep.dtype,np.integer) or len(np.unique(keep))!=len(keep):
        raise ValueError('Point selection must contain unique integer indices')
    if len(keep)<3 or keep.min()<0 or keep.max()>=len(point['point_ids']):
        raise ValueError('Invalid current-anchor selection')
    point=dict(point)
    for key in ['point_ids','anchor_xyz','anchor_uv','normal','color']:point[key]=point[key][keep].copy()
    xyz=point['anchor_xyz'];shift=(xyz.min(0)+xyz.max(0))/2;shift[2]=xyz[:,2].min()
    centered=xyz-shift;grid=np.floor(centered/voxel_size).astype(np.int64);grid-=grid.min(0)
    _,representatives,inverse=np.unique(grid,axis=0,return_index=True,return_inverse=True)
    features=np.concatenate([centered,point['color'].astype(np.float32)/255,point['normal']],axis=1)
    point.update(coord_shift=shift,coord=centered[representatives],feat=features[representatives],
                 grid_coord=grid[representatives],original_to_voxel=inverse,voxel_representatives=representatives)
    return point
