"""Prepare a separate local Point/FK runtime from supplied source and baseline assets.

Does not install packages, download assets, or copy checkpoint tensors.
Run using the existing Cosmos Python environment.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def apply_compatibility_patch(out, patch):
    # --batch alone guesses a reversed patch and silently undoes an existing
    # installation. Never allow that inference during a forward application.
    command = ['patch', '--batch', '-p1', '-d', str(out)]
    trial = subprocess.run(command + ['--forward', '--dry-run'],
                           input=patch, text=True, capture_output=True)
    if trial.returncode == 0:
        subprocess.run(command + ['--forward'], input=patch, text=True, check=True)
        return
    reverse = subprocess.run(command + ['--dry-run', '-R'],
                             input=patch, text=True, capture_output=True)
    if reverse.returncode:
        raise RuntimeError('Runtime source differs from the supported compatibility patch; ' + trial.stdout)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--baseline-bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assets,bundle,out=(x.resolve() for x in (a.assets,a.baseline_bundle,a.output))
    source=out/'source';out.mkdir(parents=True,exist_ok=True)
    if not source.exists():
        shutil.copytree(assets/'source_current',source)
    patch=(ROOT/'policy/Cosmos/bundle_compat.patch').read_text()
    apply_compatibility_patch(out, patch)
    for relative in ['cosmos_framework/inference/robot_policy/bench2dex.py',
                     'cosmos_framework/utils/bench2dex_contract.py']:
        target=source/relative
        if not target.exists():
            shutil.copy2(bundle/'source'/relative,target)
    subprocess.run([sys.executable,str(ROOT/'tools/prepare_cosmos_dcp_compat.py'),
                    '--source',str(assets/'checkpoint/iter_000003000/model'),
                    '--destination',str(out/'dcp_compat')],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/prepare_cosmos_pointfk_assets.py'),
                    '--assets',str(assets),'--output',str(out)],check=True)
    text=(assets/'config/config.yaml').read_text().replace('/data/shichaojian/models/cosmos3-edge-droid',
                                                         str(bundle/'model_assets/cosmos3-edge-droid'))
    config=yaml.safe_load(text)
    config['model']['config']['compile']['enabled']=False
    (out/'config.local.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    options=json.loads((bundle/'deployment.local.json').read_text())
    options.update(manifest=str(bundle/'metadata/manifest.json'),runtime_joint_names=None)
    (out/'deployment.json').write_text(json.dumps(options,indent=2)+'\n')
    policy=dict(policy_name='Cosmos',backend='pointfk_bundle',contract_path=str(out/'contract.json'),
        worldact_root=str(source),checkpoint_path=str(out/'dcp_compat'),training_config_path=str(out/'config.local.yaml'),
        bundle_deployment=str(out/'deployment.json'),model_environment=str(out/'model_environment.json'),
        output_dir=str(out/'inference'),execute_steps=16,num_steps=4,guidance=3)
    (out/'policy.yaml').write_text(yaml.safe_dump(policy,sort_keys=False))
    report=json.loads((out/'asset_check.json').read_text())
    report.update(status='runtime_prepared',missing=[],source=str(source),
                  limitations=['Preparation does not run inference.',
                               'Provided source is a current working-tree snapshot, not a launch-time snapshot.'])
    (out/'asset_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(out/'policy.yaml')


if __name__=='__main__':
    main()
