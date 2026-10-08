"""One recorded current snapshot through the real Point/FK RPC model; no future data."""
import argparse,json,sys,time
from pathlib import Path
import cv2,h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from policy.Cosmos.live_geometry import CurrentFrameGeometry
from policy.Cosmos.anchor_selection import hand_guided_fps
from utils.sim_query_mask import SimMaskRenderer
from script.policy_rpc import RemotePolicyClient
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=9001);p.add_argument('--frame',type=int,default=1);a=p.parse_args()
work=ROOT/'outputs/cosmos_local/pointfk_3000'
with h5py.File(ROOT/'outputs/sim_pointflow/task21_replay100/rgbd/episode_000000.hdf5') as f:
 r=SimMaskRenderer(f,ROOT.parent/'dex2bench_dataset',ROOT/'scenes/21_condiment_box_loading.yaml',fast=True)
 _,_,mask,bgr,depth=r.render(a.frame,grid_step=2)
 geom=CurrentFrameGeometry(ROOT.parent/'dex2bench_dataset/Robots_p/ur5+wuji/urdf/Multi_UR5_wuji_with_flange.urdf',selector=lambda x,k:hand_guided_fps(x,k,1024,.05))
 q=f['robot/qpos'][a.frame]; names=[n.decode() for n in f['robot/joint_names'][:]]
 g=geom.compute(rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),depth_m=depth,visible_mask=mask,intrinsic=r.K,world_from_camera=f['cameras/cam_overhead/extrinsic_world_from_cam'][a.frame],joint_names=names,qpos=q,world_from_robot=r.base,frame_id=a.frame,sim_time_sec=a.frame/20)
 images={}
 for name in ['cam_overhead','cam_wrist_left','cam_wrist_right']:
  if name in f['cameras']:
   images[name]={'rgb':cv2.cvtColor(cv2.imdecode(f[f'cameras/{name}/rgb'][a.frame],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)}
  else:
   path=ROOT/'exports/bench2dex_task21_replay100_pointfk_fullseq_v3/raw_data/bench2dex_task21/episode_000000/videos'/({'cam_wrist_left':'left_wrist.mp4','cam_wrist_right':'right_wrist.mp4'}[name])
   cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame);ok,im=cap.read();cap.release();assert ok,path
   images[name]={'rgb':cv2.cvtColor(im,cv2.COLOR_BGR2RGB)}
 contract=json.loads((work/'contract.json').read_text())
 obs=dict(joint_names=names,joint_action=dict(vector=q),observation=images,available_camera_ids=list(images),language=json.loads((ROOT/'outputs/cosmos_local/cosmos_task21_inference_bundle/metadata/manifest.json').read_text())['task_text'],current_geometry=g,geometry_frame_id=a.frame)
client=RemotePolicyClient('127.0.0.1',a.port,timeout_s=600)
t=time.perf_counter();result=client.get_action(obs);elapsed=time.perf_counter()-t
np.save(work/'smoke_actions.npy',result)
report=dict(frame=a.frame,action_shape=list(np.shape(result)),finite=bool(np.isfinite(result).all()),rpc_seconds=elapsed,geometry_timings=g['timings'])
(work/'rpc_smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
