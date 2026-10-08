"""Exercise compiled policy on changing recorded observations before live rollout."""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def recorded_observation(run,index):
    import cv2,h5py,numpy as np
    from policy.Cosmos.live_geometry import point_anchor
    files=sorted((run/'predictions').glob('*/*.npz'))
    with np.load(files[index]) as a:
        meta=json.loads(str(a['metadata_json']));frame=round(meta['sim_time_sec']*20)
        with h5py.File(next((run/'episodes').glob('*/*.hdf5'))) as f:
            names=[x.decode() for x in f['robot/joint_names'][:]];q=f['robot/qpos'][frame]
            images={c:{'rgb':cv2.cvtColor(cv2.imdecode(f[f'cameras/{c}/rgb'][frame],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)} for c in ['cam_overhead','cam_wrist_left','cam_wrist_right']}
        point=point_anchor(a['pointflow_anchor_xyz'],a['pointflow_anchor_uv'],images['cam_overhead']['rgb'],a['intrinsic'])
        fk=dict(anchor_xyz=a['fk_anchor_xyz'],anchor_uv=a['fk_anchor_uv'],point_ids=a['fk_point_ids'],image_size_wh=a['image_size_wh'])
    manifest=json.loads((ROOT/'outputs/cosmos_local/cosmos_task21_inference_bundle/metadata/manifest.json').read_text())
    return dict(joint_names=names,joint_action=dict(vector=q),observation=images,available_camera_ids=list(images),language=manifest['task_text'],geometry_frame_id=frame,current_geometry=dict(pointflow_inputs=point,fk_inputs=fk,metadata=dict(frame_id=frame,sim_time_sec=frame/20,coordinate='fixed_optical_camera',units='metre',uses_future_frames=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--compile-mode', default='default');p.add_argument('--indices',default='0,1,2,3,4,5,10,20,40,61');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np,torch,yaml
    from policy.Cosmos.pointfk_policy import PointFKPolicy
    config=yaml.safe_load((ROOT/'outputs/cosmos_local/pointfk_3000/live_episode_exec32_same_seed/policy.yaml').read_text());config.update(output_dir=str(a.output.resolve()/'model'),prediction_dir=str(a.output.resolve()/'predictions'),compile_transformer=True,compile_dynamic=True,compile_mode=a.compile_mode)
    (a.output/'policy.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    policy=PointFKPolicy(config);policy.reset(102100000)
    observations=[recorded_observation(ROOT/'outputs/cosmos_local/pointfk_3000/live_episode_async32_prefetch16',int(i)) for i in a.indices.split(',')]
    report=dict(scope='changing recorded anchors/RGB/qpos; normals re-estimated on saved selected points, not bit-identical to original dense-normal inputs',calls=[])
    try:
        for i,obs in enumerate(observations):
            torch.cuda.synchronize();t=time.perf_counter();actions=policy.get_action(copy.deepcopy(obs));torch.cuda.synchronize()
            assert actions.shape==(32,52) and np.isfinite(actions).all()
            row=dict(index=i,frame=obs['geometry_frame_id'],seconds=time.perf_counter()-t,
                     compiled_active=policy.compiled_forward.failure is None,
                     fallback=policy.compiled_forward.failure)
            report['calls'].append(row);(a.output/'report.json').write_text(json.dumps(report,indent=2));print('CALL',json.dumps(row),flush=True)
            assert row['compiled_active'], 'Compiler fell back to eager; this is not a successful compilation test'
    except Exception as e:
        report['error']=str(e);(a.output/'error.txt').write_text(traceback.format_exc());raise
    finally:(a.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
