"""Profile a recorded task21 observation without simulator/RPC or model changes."""
import argparse
from collections import defaultdict
from copy import deepcopy
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def observation(run):
    import cv2,h5py,numpy as np
    from policy.Cosmos.live_geometry import point_anchor,subset_anchor
    from policy.Cosmos.anchor_selection import hand_guided_fps
    from utils.rgbd_pointflow import lift_depth
    with np.load(run/'metrics/first_live_geometry.npz') as snap:
        rgb=snap['rgb'];depth=snap['depth'];mask=snap['mask'];K=snap['intrinsic']
        yy,xx=np.mgrid[0:depth.shape[0]:2,0:depth.shape[1]:2]
        valid=np.isin(mask[yy,xx],[2,3])&np.isfinite(depth[yy,xx])&(depth[yy,xx]>0)
        uv=np.c_[xx[valid],yy[valid]].astype(np.float32);z=depth[yy[valid],xx[valid]]
        _,valid,xyz=lift_depth(uv,depth,K)
        point=point_anchor(xyz[valid],uv[valid],rgb,K)
        keep,detail=hand_guided_fps(point['anchor_xyz'],snap['fk_xyz'],1024,.05)
        point=subset_anchor(point,keep)
        np.testing.assert_allclose(point['anchor_xyz'],snap['point_xyz'],atol=1e-6)
        fk=dict(anchor_xyz=snap['fk_xyz'],anchor_uv=snap['fk_uv'],point_ids=np.arange(42,dtype=np.int64),image_size_wh=np.array([640,480]))
        geometry=dict(pointflow_inputs=point,fk_inputs=fk,metadata=dict(frame_id=0,sim_time_sec=0.,coordinate='fixed_optical_camera',units='metre',uses_future_frames=False))
    episode=next((run/'episodes').glob('*/*.hdf5'))
    with h5py.File(episode) as f:
        names=[x.decode() for x in f['robot/joint_names'][:]];q=f['robot/qpos'][0]
        images={name:{'rgb':cv2.cvtColor(cv2.imdecode(f[f'cameras/{name}/rgb'][0],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)} for name in ['cam_overhead','cam_wrist_left','cam_wrist_right']}
        images['cam_overhead']['rgb']=rgb
    manifest=json.loads((ROOT/'outputs/cosmos_local/cosmos_task21_inference_bundle/metadata/manifest.json').read_text())
    return dict(joint_names=names,joint_action=dict(vector=q),observation=images,available_camera_ids=list(images),language=manifest['task_text'],current_geometry=geometry,geometry_frame_id=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',type=Path,default=ROOT/'outputs/cosmos_local/pointfk_3000/live_episode_exec32_same_seed');p.add_argument('--repeats',type=int,default=5);p.add_argument('--trace',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import yaml,numpy as np,torch
    from policy.Cosmos.pointfk_policy import PointFKPolicy
    config=yaml.safe_load(a.config.read_text());config.update(output_dir=str(a.output.resolve()/'model'),prediction_dir=str(a.output.resolve()/'predictions'))
    (a.output/'policy.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    events=[]
    def mark(name):
        events.append(dict(name=name,epoch=time.time()));(a.output/'events.json').write_text(json.dumps(events,indent=2))
    log=(a.output/'gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv,nounits','-lms','100'],stdout=log,stderr=subprocess.STDOUT)
    try:
        obs=observation(a.run);mark('load_start');policy=PointFKPolicy(config);policy.reset(102100000);mark('load_end')
        def call():
            torch.cuda.synchronize();t=time.perf_counter();out=policy.get_action(deepcopy(obs));torch.cuda.synchronize()
            assert out.shape==(32,52) and np.isfinite(out).all()
            return time.perf_counter()-t
        warm=[];plain=[]
        for i in range(2):
            mark(f'warmup_{i}_start');warm.append(call());mark(f'warmup_{i}_end')
        for i in range(a.repeats):
            mark(f'plain_{i}_start');plain.append(call());mark(f'plain_{i}_end')
        phases=defaultdict(list);originals=[];active={'enabled':True,'sync':True};shapes={}
        def wrap(obj,method,label):
            original=getattr(obj,method)
            @functools.wraps(original)
            def measured(*args,**kwargs):
                if not active['enabled']:return original(*args,**kwargs)
                if active['sync']:torch.cuda.synchronize()
                started=time.perf_counter()
                with torch.profiler.record_function(label):
                    result=original(*args,**kwargs)
                if active['sync']:torch.cuda.synchronize()
                phases[label].append(time.perf_counter()-started)
                if label=='transformer' and args:
                    def shape(v):
                        if isinstance(v,torch.Tensor):return list(v.shape)
                        if isinstance(v,dict):return {k:shape(x) for k,x in v.items()}
                        return type(v).__name__
                    shapes['transformer_input']=shape(args[0])
                return result
            setattr(obj,method,measured);originals.append((obj,method,original))
        model=policy.backend.model;net=model.net
        for obj,method,label in [(policy.backend,'_build_batch','build_batch'),(model,'get_data_and_condition','data_condition'),(model,'encode','video_vae_encode'),(model,'denoise','denoise'),(net,'forward','network_forward'),(net.language_model,'forward','transformer'),(net.pointflow_branch,'encode','point_encode_total'),(net.pointflow_branch.geometry,'forward','sonata'),(net.pointflow_branch,'decode','point_decode'),(net.fk_branch,'encode','fk_encode'),(net.fk_branch,'decode','fk_decode')]:wrap(obj,method,label)
        per_call=[]
        for i in range(3):
            phases.clear();mark(f'phases_{i}_start');elapsed=call();mark(f'phases_{i}_end')
            per_call.append(dict(seconds=elapsed,phases={k:dict(calls=len(v),seconds=sum(v)) for k,v in phases.items()}))
        report=dict(scope='same recorded snapshot; model only, no Isaac/RPC; synchronized nested phases overlap and must not be summed',frame=0,source_run=str(a.run),warmup_seconds=warm,plain_seconds=plain,phase_runs=per_call,shapes=shapes,gpu=torch.cuda.get_device_name(),torch=torch.__version__,peak_allocated_bytes=torch.cuda.max_memory_allocated(),config=config)
        (a.output/'profile.json').write_text(json.dumps(report,indent=2));print('PROFILE',json.dumps(report),flush=True)
        if a.trace:
            active['sync']=False;mark('trace_start')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:call()
            mark('trace_end');prof.export_chrome_trace(str(a.output/'trace.json'))
            (a.output/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
        for obj,method,original in reversed(originals):setattr(obj,method,original)
    finally:
        mark('end');monitor.terminate();monitor.wait();log.close()

if __name__=='__main__':main()
