"""Numerically compare training transforms against the proposed inference batch."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import typing
import typing_extensions

if not hasattr(typing, 'override'):
    typing.override = typing_extensions.override
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.bundle.resolve()/'source'))
import numpy as np
import torch
from cosmos_framework.inference.robot_policy.bench2dex import Bench2DexPolicy
from cosmos_framework.utils.bench2dex_contract import CAMERAS, VIEW_DESCRIPTION, compose_rgb
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.action.action_processing import ActionProcessor
manifest = json.loads((a.bundle/'metadata/manifest.json').read_text())
# No checkpoint or CUDA: exercise the actual two independently implemented paths.
policy = object.__new__(Bench2DexPolicy)
policy.device = 'cpu'
policy.view_width = manifest['view_width']
policy.input_video_key = 'video'
policy.config = SimpleNamespace(model=SimpleNamespace(native_chunk_size=32,max_action_dim=64,
    native_action_dim=52,task=manifest['task_text'],resolution='480',domain_name='bench2dex_wuji'),
    deployment=SimpleNamespace(action_rate_hz=20.0))
rng = np.random.default_rng(923)
images = {c:rng.integers(0,256,(480,640,3),dtype=np.uint8) for c in CAMERAS}
state = rng.uniform(-1,1,52).astype(np.float32)
batch = policy._build_batch(images, state)
video = torch.zeros(3,33,720,640,dtype=torch.uint8)
video[:,0] = torch.from_numpy(compose_rgb(images))
action = torch.zeros(33,52)
action[0] = torch.from_numpy(state)
sample = dict(ai_caption=manifest['task_text'],video=video,action=action,
              conditioning_fps=torch.tensor(20),mode='wam',viewpoint='concat_view',
              additional_view_description=VIEW_DESCRIPTION,domain_id=torch.tensor(28))
transform = ActionTransformPipeline(tokenizer_config=None,cfg_dropout_rate=0,
    max_action_dim=64,format_prompt_as_json=True)
train = transform(copy.deepcopy(sample),'480',action_normalizer=None)
report = dict(video_exact=bool(torch.equal(train['video'],batch['video'][0][0])),
    action_exact=bool(torch.equal(train['action'],batch['action'][0][0])),
    prompt_exact=json.dumps(train['ai_caption'])==batch['ai_caption'][0],
    image_size_exact=bool(torch.equal(train['image_size'],batch['image_size'][0])),
    sequence_plan_exact=train['sequence_plan']==batch['sequence_plan'][0],
    resized_video_shape=list(train['video'].shape),model_executed=False,
    scope='Current supplied training source versus deployment; not proof of historical training version')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not all(report[k] for k in report if k.endswith('_exact')):
    raise SystemExit('Training and inference disagree')
