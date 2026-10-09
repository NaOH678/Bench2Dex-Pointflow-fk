# WorldAct runtime snapshot

This directory contains the WorldAct/Cosmos source used by the Bench2Dex
PointFlow/FK inference adapter. It includes the supplied training working-tree
changes and local Torch/FlashAttention compatibility changes. It is not an
unmodified upstream release. See `SOURCE_MANIFEST.json` for provenance and hashes,
and `LICENSE`, `NOTICE`, and `ATTRIBUTIONS.md` for upstream licensing.

`cosmos_framework/` and `packages/` are copied as a unit because inference imports
configuration, processor, model and checkpoint classes across these packages.
Training data, model weights, generated outputs, caches and upstream example media
are excluded. Existing compatibility patches are already applied: do not apply
`policy/Cosmos/bundle_compat.patch` again to this snapshot.

See `deployment/cosmos/SIM_INFERENCE.md` at the Bench2Dex repository root.
