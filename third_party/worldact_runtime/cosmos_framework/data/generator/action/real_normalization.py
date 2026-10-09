"""Training-statistics contract and reversible real-robot action scaling."""

import hashlib
import json
from pathlib import Path

import numpy as np


class RealActionNormalizer:
    def __init__(self, path, *, expected_sha256=None):
        payload = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError("Real action statistics SHA256 mismatch")
        self.stats = s = json.loads(payload)
        if (
            s["schema"] != "singlerighthand_action_quantile_v1"
            or s["fit_split"] != "train"
            or s["forward_clamp"] is not None
        ):
            raise ValueError("Invalid real action statistics contract")
        self.offset = np.asarray(s["offset"], np.float32)
        self.scale = np.asarray(s["scale"], np.float32)
        if (
            self.offset.shape != (27,)
            or self.scale.shape != (27,)
            or not np.isfinite([self.offset, self.scale]).all()
            or (self.scale <= 0).any()
        ):
            raise ValueError("Invalid real action offset/scale")

    def normalize(self, x):
        return (x - self.offset) / self.scale

    def denormalize(self, x):
        """Accept numpy or torch arrays, preserving any padded channels."""
        if x.shape[-1] < 27:
            raise ValueError("Expected at least 27 action dimensions")
        if hasattr(x, "new_tensor"):
            y = x.clone()
            y[..., :27] = x[..., :27] * x.new_tensor(self.scale) + x.new_tensor(self.offset)
        else:
            y = np.array(x, copy=True)
            y[..., :27] = x[..., :27] * self.scale + self.offset
        return y

    def validate_dataset(
        self, *, cache_root, all_episodes, order, num_val, split_seed, split_val_ratio, action_names, arm_action_space
    ):
        s = self.stats
        if (
            s["split_seed"] != split_seed
            or s["split_val_ratio"] != split_val_ratio
            or s["action_names"] != action_names
            or s["arm_action_space"] != arm_action_space
        ):
            raise ValueError("Real action statistics split/action-space mismatch")
        names = [all_episodes[i].name for i in order[num_val:]]
        if names != s["train_episodes"] or [all_episodes[i].name for i in order[:num_val]] != s["val_episodes"]:
            raise ValueError("Real action statistics episode split mismatch")
        root = Path(cache_root)
        if hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() != s["manifest_sha256"]:
            raise ValueError("Real action statistics manifest mismatch")
        for name in names:
            if (
                hashlib.sha256((root / "episodes" / f"{name}.npz").read_bytes()).hexdigest()
                != s["episode_sha256"][name]
            ):
                raise ValueError(f"Real action statistics source changed: {name}")
