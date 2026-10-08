"""Compare sampling settings on identical observations; no action smoothing."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import h5py
import numpy as np
from policy.Cosmos.deploy_policy import get_model
from policy.Cosmos.contract import CAMERAS

out=Path('outputs/cosmos_local/jitter_diagnosis');out.mkdir(parents=True,exist_ok=True)
config=json.loads(Path('outputs/cosmos_local/deploy_baseline.json').read_text())
config['execute_steps']=32
config['output_dir']=str(out/'model')
model=get_model(config)
results=[]
with h5py.File('exports/bench2dex_task21_pointfk10_rgb3_v2/raw_data/bench2dex_task21/episode_000000/observations.hdf5') as f:
 names=[x.decode() for x in f['robot/joint_names'][:]]
 for frame in [1,200,400]:
  obs=dict(joint_names=names,available_camera_ids=list(CAMERAS),joint_action=dict(vector=f['robot/qpos'][frame]),language=f['meta/instruction'][()].decode(),observation={})
  for camera in CAMERAS:
   obs['observation'][camera]={'rgb':cv2.cvtColor(cv2.imdecode(f[f'cameras/{camera}/rgb'][frame],1),cv2.COLOR_BGR2RGB)}
  for steps in [4,16]:
   model.backend.config.model.num_steps=steps
   model.reset(42)
   a=model.get_action(obs);np.save(out/f'frame{frame}_steps{steps}.npy',a)
   d=np.diff(a,axis=0);dd=np.diff(d,axis=0)
   significant=(abs(d[:-1])>.003)&(abs(d[1:])>.003)
   results.append(dict(frame=frame,sampling_steps=steps,mean_abs_step_rad=float(abs(d).mean()),mean_abs_second_diff_rad=float(abs(dd).mean()),first4_second_diff_rad=float(abs(np.diff(a[:4],n=2,axis=0)).mean()),reversal_rate=float((d[:-1]*d[1:]<0)[significant].mean())))
report=dict(scope='Same observation and seed; open-loop sampling diagnosis, not a closed-loop success test',results=results)
(out/'sampling_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
