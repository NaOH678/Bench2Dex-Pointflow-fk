"""Check current-frame conditioning against training exports and time CPU preparation.

Recorded snapshots validate geometry; these timings exclude live rendering/RPC/model inference.
"""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT.parent/'WorldAct-pointflow-fk')]
import cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation
from policy.Cosmos.live_geometry import point_anchor,CurrentFrameGeometry
from utils.sim_surface_gt import transform
from utils.sim_query_mask import SimMaskRenderer
from cosmos_framework.data.pointflow_window import prepare_window,PointFlowTiming
root=ROOT/'exports/bench2dex_task21_replay100_pointfk_fullseq_v3'
ep=root/'pf_out/bench2dex_task21/labeled/episode_000000'
out=ROOT/'outputs/cosmos_local/pointfk_live';out.mkdir(parents=True,exist_ok=True)
flat={k:np.load(ep/(k+'.npy'),mmap_mode='r') for k in ['position','uv_px','valid']}
video=cv2.VideoCapture(str(root/'raw_data/bench2dex_task21/episode_000000/videos/head.mp4'))
with h5py.File(ROOT/'outputs/sim_pointflow/task21_replay100/rgbd/episode_000000.hdf5') as f:
 K=f['cameras/cam_overhead/intrinsic'][:]
 for t in [1,32,348,674]:
  video.set(cv2.CAP_PROP_POS_FRAMES,t);ok,bgr=video.read();assert ok
  xyz=flat['position'][t].copy();xyz[~flat['valid'][t]]=np.nan
  a=point_anchor(xyz,flat['uv_px'][t],cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),K)
  b=prepare_window(ep,t,max_points=16384,timing=PointFlowTiming(fps=20,steps=32,steps_per_token=4),anchor_frame_bgr=bgr)
  for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-6,rtol=1e-6,err_msg=k)
  print('ANCHOR_MATCH',t,len(a['point_ids']),flush=True)
 video.release()
 renderer=SimMaskRenderer(f,ROOT.parent/'dex2bench_dataset',ROOT/'scenes/21_condiment_box_loading.yaml',fast=True)
 live=CurrentFrameGeometry(ROOT.parent/'dex2bench_dataset/Robots_p/ur5+wuji/urdf/Multi_UR5_wuji_with_flange.urdf')
 names=[x.decode() for x in f['robot/joint_names'][:]]
 base=transform(Rotation.from_euler('z',90,degrees=True).as_matrix(),[.5,-.43,.75])
 fk=np.load(root/'raw_data/bench2dex_task21/episode_000000/annotations/wuji_fk21.npz')['positions_camera']
 results=[]
 for t in [1,32,64,160,250,348,450,550,650,674]:
  begin=time.perf_counter();_,_,mask,bgr,depth=renderer.render(t,grid_step=2);mask_sec=time.perf_counter()-begin
  sample=live.compute(rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),depth_m=depth,visible_mask=mask,intrinsic=K,
      world_from_camera=f['cameras/cam_overhead/extrinsic_world_from_cam'][t],joint_names=names,qpos=f['robot/qpos'][t],
      world_from_robot=base,frame_id=t,sim_time_sec=t/20)
  np.testing.assert_allclose(sample['fk_inputs']['anchor_xyz'],fk[t].reshape(42,3),atol=1e-6,rtol=1e-6)
  row=dict(frame=t,points=len(sample['pointflow_inputs']['point_ids']),mask_sec=mask_sec,**sample['timings'])
  results.append(row);print(json.dumps(row),flush=True)
report=dict(status='passed',mask_backend='scalar_fast',anchor_matches_training=True,fk_matches_training=True,samples=results,
 limitations=['Recorded snapshots only; live camera rendering, RPC and point/FK model sampling have not been timed.'])
(out/'geometry_check_fast.json').write_text(json.dumps(report,indent=2))
