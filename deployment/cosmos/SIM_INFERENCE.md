# Portable simulation inference

The `sim_inference` branch contains both the Bench2Dex inference adapter and the
matching WorldAct runtime under `third_party/worldact_runtime/`. A separate
WorldAct checkout is not required. The default templates match the supplied
Task21 UR5+Wuji PF/FK training run; another training run needs its own matching
contract, normalization, model configuration and PF/FK settings.

## External files

Prepare these on the target machine; they are not Git files:

- The selected checkpoint's complete `iter_.../model/` DCP directory, including
  `.metadata` and all `.distcp` shards.
- `cosmos3-edge-droid/` model assets: tokenizer files and `vae/Wan2.2_VAE.pth`.
- Matching `sonata_small.pth` weights.
- Bench2Dex robot USD/URDF/meshes, scene/object assets and initialization episodes
  when running simulation. Existing code uses `../dex2bench_dataset/` by default.
- Compatible Cosmos and IsaacLab Python environments. See `README.md` in this
  directory for the tested dependencies; environments are not copied through Git.

## Prepare local paths

Run from the repository root using the existing inference Python environment:

```bash
PY=/path/to/inference/python
"$PY" tools/prepare_sim_inference.py \
  --checkpoint /path/to/iter_000016000/model \
  --model-assets /path/to/cosmos3-edge-droid \
  --sonata /path/to/sonata_small.pth \
  --output outputs/pointfk_16000
```

This generates local absolute paths and DCP compatibility metadata. Tensor shards
are symlinks to the supplied checkpoint; no weights are downloaded or copied.
Use a new output directory for each checkpoint. Generated configuration defaults
to synchronous prediction/execution of 32 steps, 4 denoising steps, no smoothing,
no Transformer compilation and original frame preprocessing.

```bash
COSMOS_PYTHON="$PY" \
COSMOS_CONFIG="$PWD/outputs/pointfk_16000/policy.yaml" \
COSMOS_RUN_DIR="$PWD/outputs/pointfk_16000/episode_test" \
BASELINE_ANCHOR_HDF5=/path/to/episode_000000.hdf5 \
  bash tools/run_cosmos_local_sim.sh \
  --live-pointfk --seed 100000000 --episode-steps 827 --early-stop
```

The launcher currently starts a single-GPU model and simulator using port 9000.
Setting several visible GPUs does not enable model parallelism. Concurrent
multi-GPU episode workers need separate GPU assignments, ports and output paths;
this branch does not yet implement that launcher. Asynchronous prefetch is
optional and does not establish real-time control.
