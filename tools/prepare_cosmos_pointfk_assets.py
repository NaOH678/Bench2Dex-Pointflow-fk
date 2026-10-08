"""Prepare local Point/FK checkpoint paths and explicit training contracts; no model launch."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from torch.distributed.checkpoint import FileSystemReader
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assets=a.assets.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
norm_path=assets/'normalization/normalization_train_only.json';norm=json.loads(norm_path.read_text())
selection=json.loads((assets/'pointflow/manifest.json').read_text())
assert selection['schema']=='sim_pointfk_handguided1024_v1' and not selection['selection_uses_future']
assert len(norm['joint_names'])==52 and norm['state_transform']=='same_as_action'
low,high=[np.asarray(norm['action'][k],np.float32) for k in ['q01','q99']]
np.testing.assert_allclose(norm['offset'],(low+high)/2,atol=1e-7)
np.testing.assert_allclose(norm['scale'],np.maximum((high-low)/2,.05),atol=1e-7)
assert hashlib.sha256(norm_path.read_bytes()).hexdigest()==selection['action_stats_sha256']
contract=dict(action_representation='absolute_joint_position_radians',
    view_composition='head_top__left_wrist_bottom_left__right_wrist_bottom_right',
    additional_view_description='Head on top; left wrist bottom-left; right wrist bottom-right.',
    fps=20,action_chunk_size=32,resolution='480',domain_name='bench2dex_wuji')
contract['joint_names']=norm['joint_names'];contract['normalization']=dict(kind='affine',offset=norm['offset'],scale=norm['scale'],forward_clamp=norm['forward_clamp'])
contract['geometry']=dict(selection={k:v for k,v in selection.items() if k not in ['windows','split']},fk_keypoints=42,
                        coordinate='fixed_optical_camera',units='metre',requires_current_frame=True,training_code_verified=False)
(out/'contract.json').write_text(json.dumps(contract,indent=2))
env=dict(POINTFLOW_SONATA_CHECKPOINT=str(assets/'weights/sonata_small.pth'),POINTFLOW_TOKEN_MODE='cluster',POINTFLOW_SONATA_STAGE='2',POINTFLOW_FREEZE_SONATA='false',POINTFLOW_CLUSTER_TOKEN_CAP='1024',POINTFLOW_DECODE_SKIP_LEVELS='1,2',POINTFLOW_DECODE_POINT_BLOCKS='4',POINTFLOW_GEOMETRY_MOTION_FUSION='false',POINTFLOW_FPS='20',POINTFLOW_STEPS='32',POINTFLOW_STEPS_PER_TOKEN='4',FK_ENCODER_CHECKPOINT='1',FK_KEYPOINTS='42',FK_FPS='20',FK_STEPS='32',FK_STEPS_PER_TOKEN='4',POINTFLOW_FK_LOCAL_ROPE='true',POINTFLOW_FK_ATTN_IMPL='partition',COSMOS_FLASH2_VARLEN='1',I4_ATTN_BACKENDS='flash2',POINTFLOW_REFERENCE_ATTENTION='false',POINTFLOW_DISPLACEMENT_SCALE=str(selection['train_only_scales']['pointflow']),FK_DISPLACEMENT_SCALE=str(selection['train_only_scales']['fk']),POINTFLOW_DISPLACEMENT_FRAME_SCALES='',POINTFLOW_DISPLACEMENT_FRAME_SCALES_FILE='')
(out/'model_environment.json').write_text(json.dumps(env,indent=2))
m=FileSystemReader(out/'dcp_compat').read_metadata();keys=list(m.state_dict_metadata)
counts={prefix:sum(k.startswith(prefix) for k in keys) for prefix in ['net.','net_ema.','net.pointflow_branch.','net.fk_branch.']}
assert all(counts.values())
has_source=(assets/'source_current/cosmos_framework').is_dir()
report=dict(status='assets_checked' if has_source else 'assets_checked_pending_training_source',assets=str(assets),checkpoint=str(out/'dcp_compat'),checkpoint_tensor_counts=counts,selection_points=selection['points'],fk_points=42,normalization_verified=True,missing=[] if has_source else ['Training working-tree changes relative to e334e73'],limitations=['This preparation script does not run model inference; see separate runtime validation reports.'])
(out/'asset_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
