"""Prepare portable PF/FK inference configuration using the committed runtime.

Uses existing checkpoint/model assets; does not install or download anything.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='DCP model directory containing .metadata')
    parser.add_argument('--model-assets', type=Path, required=True, help='cosmos3-edge-droid directory containing tokenizer and vae/')
    parser.add_argument('--sonata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--templates', type=Path, default=ROOT / 'deployment/cosmos/pointfk_task21',
                        help='Contract/config from the matching training run; defaults to the supplied Task21 PF/FK run')
    parser.add_argument('--execute-steps', type=int, choices=range(1, 33), default=32)
    args = parser.parse_args()
    checkpoint, assets, sonata, out, templates = (
        p.resolve() for p in (args.checkpoint, args.model_assets, args.sonata, args.output, args.templates))
    for p in (checkpoint / '.metadata', assets / 'vae/Wan2.2_VAE.pth', sonata):
        if not p.is_file():
            parser.error(f'Missing external asset: {p}')
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory to avoid mixing checkpoints/configurations')
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/prepare_cosmos_dcp_compat.py'),
                    '--source', str(checkpoint), '--destination', str(out / 'dcp_compat')], check=True)
    for name in ('contract.json', 'manifest.json'):
        shutil.copyfile(templates / name, out / name)
    environment = json.loads((templates / 'model_environment.json').read_text())
    environment['POINTFLOW_SONATA_CHECKPOINT'] = str(sonata)
    (out / 'model_environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    text = (templates / 'config.template.yaml').read_text()
    for key, value in {'COSMOS_MODEL_ASSETS': assets, 'BASE_DCP': checkpoint,
                       'TRAINING_DATA_UNUSED': out / 'unused_training_data',
                       'TRAINING_OUTPUT_UNUSED': out / 'unused_training_output'}.items():
        text = text.replace('${' + key + '}', str(value))
    config = yaml.safe_load(text)
    config['model']['config']['compile']['enabled'] = False
    (out / 'config.local.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    deployment = dict(manifest=str(out / 'manifest.json'), runtime_joint_names=None,
                      seed=42, use_ema_weights=True)
    (out / 'deployment.json').write_text(json.dumps(deployment, indent=2) + '\n')
    policy = dict(policy_name='Cosmos', backend='pointfk_bundle',
                  worldact_root=str(ROOT / 'third_party/worldact_runtime'),
                  checkpoint_path=str(out / 'dcp_compat'),
                  contract_path=str(out / 'contract.json'),
                  training_config_path=str(out / 'config.local.yaml'),
                  bundle_deployment=str(out / 'deployment.json'),
                  model_environment=str(out / 'model_environment.json'),
                  output_dir=str(out / 'inference'), execute_steps=args.execute_steps,
                  num_steps=4, guidance=3, seed=42, compile_transformer=False,
                  preprocess_observed_frame_only=False, action_smoothing=dict(mode='none'))
    (out / 'policy.yaml').write_text(yaml.safe_dump(policy, sort_keys=False))
    print(out / 'policy.yaml')


if __name__ == '__main__':
    main()
