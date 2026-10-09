# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_spec import DimType
from cosmos_framework.data.generator.action.datasets.dualhand_joint_cache_dataset import DualHandJointCacheDataset


def _build_dataset(tmp_path) -> tuple[DualHandJointCacheDataset, list[tuple[np.ndarray, np.ndarray]]]:
    cache_root = tmp_path / "cache"
    (cache_root / "episodes").mkdir(parents=True)
    (cache_root / "video_frames").mkdir()

    arrays = []
    episode_rows = []
    video_rows = []
    for episode_idx, num_frames in enumerate((67, 69)):
        name = f"episode_{episode_idx:04d}"
        state = np.arange(num_frames * 54, dtype=np.float32).reshape(num_frames, 54) + episode_idx * 100_000
        action = state + 10_000
        np.savez(cache_root / "episodes" / f"{name}.npz", state=state, action=action)
        frames = np.full((num_frames, 3, 6, 8), episode_idx, dtype=np.uint8)
        relative_path = f"video_frames/{name}.npy"
        np.save(cache_root / relative_path, frames)
        arrays.append((state, action))
        episode_rows.append({"name": name, "num_frames": num_frames, "source_fps": 30.0})
        video_rows.append(
            {
                "name": name,
                "path": relative_path,
                "shape": list(frames.shape),
                "image_size": [6, 8, 6, 8],
            }
        )

    (cache_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "cosmos_dualhand_joint_v1",
                "schema_version": 1,
                "dataset_type": "dualhand_joint",
                "arm_action_space": "joint",
                "task_text": "use a micropipette",
                "num_episodes": len(episode_rows),
                "total_frames": sum(row["num_frames"] for row in episode_rows),
                "state_dim": 54,
                "action_dim": 54,
                "units": {"state": "radian", "action": "radian"},
                "episodes": episode_rows,
            }
        )
    )
    (cache_root / "video_manifest.json").write_text(
        json.dumps(
            {
                "schema": "cosmos_dualhand_joint_video_v1",
                "schema_version": 1,
                "resolution": "480",
                "dtype": "uint8",
                "layout": "TCHW",
                "episodes": video_rows,
            }
        )
    )
    dataset = DualHandJointCacheDataset(
        cache_root=str(cache_root),
        split="full",
        video_cache_resolution="480",
    )
    return dataset, arrays


def test_temporal_window_joint_layout_and_cache_only_video(tmp_path) -> None:
    dataset, arrays = _build_dataset(tmp_path)

    assert len(dataset) == 8
    assert dataset.get_shuffle_blocks() == [(0, 3), (3, 5)]
    observation, action_indices = dataset._window_indices(0, 0)
    np.testing.assert_array_equal(observation, np.arange(0, 65, 2))
    np.testing.assert_array_equal(action_indices, np.arange(0, 64, 2))

    sample = dataset[0]
    assert sample["video"].shape == (3, 33, 6, 8)
    assert sample["video"].dtype == torch.uint8
    assert sample["action"].shape == (33, 54)
    torch.testing.assert_close(sample["action"][0], torch.from_numpy(arrays[0][0][0]))
    torch.testing.assert_close(sample["action"][1:], torch.from_numpy(arrays[0][1][action_indices]))
    assert int(sample["domain_id"]) == 27

    spec = dataset.action_spec
    assert dataset.arm_action_space == "joint"
    assert spec.dim == 54
    assert spec.names[0] == "left_arm_0"
    assert spec.names[7] == "left_hand_0"
    assert spec.names[27] == "right_arm_0"
    assert spec.names[34] == "right_hand_0"
    assert set(spec.types) == {DimType.JOINT}


def test_requires_dualhand_video_manifest(tmp_path) -> None:
    dataset, _ = _build_dataset(tmp_path)
    video_manifest = dataset._cache_root / "video_manifest.json"
    video_manifest.unlink()

    try:
        DualHandJointCacheDataset(cache_root=str(dataset._cache_root), split="full")
    except FileNotFoundError as exc:
        assert "requires the precomputed video cache" in str(exc)
    else:
        raise AssertionError("Expected missing video manifest to fail")


def test_requires_explicit_joint_action_space(tmp_path) -> None:
    dataset, _ = _build_dataset(tmp_path)
    manifest_path = dataset._cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("arm_action_space")
    manifest_path.write_text(json.dumps(manifest))

    try:
        DualHandJointCacheDataset(cache_root=str(dataset._cache_root), split="full")
    except ValueError as exc:
        assert "arm_action_space='joint'" in str(exc)
    else:
        raise AssertionError("Expected missing arm_action_space to fail")
