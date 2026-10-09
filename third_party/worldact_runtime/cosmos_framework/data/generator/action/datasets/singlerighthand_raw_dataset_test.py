# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from types import MethodType

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.action_spec import DimType
from cosmos_framework.data.generator.action.datasets.singlerighthand_raw_dataset import SingleRightHandRawDataset


def _build_dataset(
    tmp_path,
    *,
    use_precomputed_video: bool = False,
    arm_action_space: str | None = None,
    pointflow_manifest: str | None = None,
) -> tuple[SingleRightHandRawDataset, list[tuple[np.ndarray, np.ndarray]]]:
    raw_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    (cache_root / "episodes").mkdir(parents=True)

    arrays = []
    episode_rows = []
    for episode_idx, num_frames in enumerate((67, 69)):
        name = f"episode_{episode_idx:04d}"
        state = np.arange(num_frames * 27, dtype=np.float32).reshape(num_frames, 27) + episode_idx * 100_000
        action = state + 10_000
        np.savez(cache_root / "episodes" / f"{name}.npz", state=state, action=action)
        arrays.append((state, action))
        episode_rows.append({"name": name, "num_frames": num_frames, "source_fps": 30.0})

    manifest = {"schema_version": 2, "task_text": "make a sandwich", "episodes": episode_rows}
    if arm_action_space is not None:
        manifest["arm_action_space"] = arm_action_space
    (cache_root / "manifest.json").write_text(json.dumps(manifest))
    if use_precomputed_video:
        (cache_root / "video_frames").mkdir()
        video_rows = []
        for episode_idx, row in enumerate(episode_rows):
            frames = np.full((row["num_frames"], 3, 6, 8), episode_idx, dtype=np.uint8)
            relative_path = f"video_frames/{row['name']}.npy"
            np.save(cache_root / relative_path, frames)
            video_rows.append(
                {
                    "name": row["name"],
                    "path": relative_path,
                    "shape": list(frames.shape),
                    "image_size": [6, 8, 6, 8],
                }
            )
        (cache_root / "video_manifest.json").write_text(
            json.dumps({"schema_version": 1, "resolution": "480", "episodes": video_rows})
        )
    dataset = SingleRightHandRawDataset(
        root=str(raw_root),
        cache_root=str(cache_root),
        split="full",
        use_image_augmentation=False,
        use_precomputed_video=use_precomputed_video,
        video_cache_resolution="480" if use_precomputed_video else None,
        pointflow_manifest=pointflow_manifest,
    )

    def fake_load_video(self, episode_idx: int, observation_indices: np.ndarray) -> torch.Tensor:
        del self
        return torch.full((3, len(observation_indices), 8, 8), episode_idx, dtype=torch.uint8)

    if not use_precomputed_video:
        dataset._load_video = MethodType(fake_load_video, dataset)
    return dataset, arrays


@pytest.mark.parametrize("pointflow_manifest", [None, ""])
def test_wam_disables_pointflow_with_absent_manifest(tmp_path, pointflow_manifest) -> None:
    dataset, _ = _build_dataset(tmp_path, pointflow_manifest=pointflow_manifest)
    assert dataset._pointflow_source is None
    assert dataset[0]["action"].shape == (33, 27)


def test_official_temporal_window_and_episode_boundaries(tmp_path) -> None:
    dataset, arrays = _build_dataset(tmp_path)

    assert len(dataset) == 8
    assert dataset.get_shuffle_blocks() == [(0, 3), (3, 5)]

    observation_0, action_0 = dataset._window_indices(0, 0)
    observation_1, action_1 = dataset._window_indices(0, 1)
    np.testing.assert_array_equal(observation_0, np.arange(0, 65, 2))
    np.testing.assert_array_equal(action_0, np.arange(0, 64, 2))
    np.testing.assert_array_equal(observation_1, np.arange(1, 66, 2))
    np.testing.assert_array_equal(action_1, np.arange(1, 65, 2))

    sample_0 = dataset[0]
    sample_1 = dataset[1]
    sample_2 = dataset[2]
    assert sample_0["video"].shape == (3, 33, 8, 8)
    assert sample_0["action"].shape == (33, 27)
    torch.testing.assert_close(sample_0["action"][0], torch.from_numpy(arrays[0][0][0]))
    torch.testing.assert_close(sample_0["action"][1:], torch.from_numpy(arrays[0][1][action_0]))
    torch.testing.assert_close(sample_1["action"][0], torch.from_numpy(arrays[0][0][1]))
    torch.testing.assert_close(sample_2["action"][0], torch.from_numpy(arrays[0][0][2]))
    torch.testing.assert_close(dataset[3]["action"][0], torch.from_numpy(arrays[1][0][0]))
    assert int(sample_0["domain_id"]) == 26


def test_action_spec_follows_manifest_arm_action_space(tmp_path) -> None:
    joint_dataset, _ = _build_dataset(tmp_path / "joint", arm_action_space="joint")
    legacy_dataset, _ = _build_dataset(tmp_path / "legacy")

    assert joint_dataset.arm_action_space == "joint"
    assert joint_dataset.action_spec.dim == 27
    assert joint_dataset.action_spec.names[:8] == [
        "right_arm_0",
        "right_arm_1",
        "right_arm_2",
        "right_arm_3",
        "right_arm_4",
        "right_arm_5",
        "right_arm_6",
        "right_hand_0",
    ]
    assert set(joint_dataset.action_spec.types) == {DimType.JOINT}
    assert legacy_dataset.arm_action_space == "eef"
    assert legacy_dataset.action_spec.types[:3] == [DimType.POS] * 3
    assert legacy_dataset.action_spec.types[3:7] == [DimType.ROT] * 4


def test_compose_views_preserves_aspect_ratio() -> None:
    head = torch.ones((2, 3, 4, 8), dtype=torch.uint8)
    wrist = torch.full((2, 3, 4, 16), 2, dtype=torch.uint8)

    composed = SingleRightHandRawDataset._compose_views(head, wrist)

    assert composed.shape == (2, 3, 6, 8)
    assert torch.all(composed[:, :, :2] == 2)
    assert torch.all(composed[:, :, 2:] == 1)


def test_load_video_reuses_composed_frames_across_sliding_windows(tmp_path) -> None:
    dataset, _ = _build_dataset(tmp_path)

    class FakeReader:
        def __init__(self, width: int, value: int) -> None:
            self.width = width
            self.value = value
            self.calls: list[np.ndarray] = []

        def get_frames(self, indices: np.ndarray) -> torch.Tensor:
            self.calls.append(indices.copy())
            return torch.full((len(indices), 3, 4, self.width), self.value, dtype=torch.uint8)

    readers = {
        "head": FakeReader(width=8, value=1),
        "right_wrist": FakeReader(width=16, value=2),
    }

    def fake_get_reader(self, episode_idx: int, camera: str) -> FakeReader:
        del self, episode_idx
        return readers[camera]

    dataset._get_reader = MethodType(fake_get_reader, dataset)
    first = np.arange(0, 65, 2, dtype=np.int64)
    second = np.arange(2, 67, 2, dtype=np.int64)

    first_video = SingleRightHandRawDataset._load_video(dataset, 0, first)
    second_video = SingleRightHandRawDataset._load_video(dataset, 0, second)

    assert first_video.shape == second_video.shape == (3, 33, 6, 8)
    np.testing.assert_array_equal(readers["head"].calls[0], first)
    np.testing.assert_array_equal(readers["head"].calls[1], np.asarray([66]))
    np.testing.assert_array_equal(readers["right_wrist"].calls[1], np.asarray([66]))


def test_load_video_eviction_never_drops_frames_needed_now(tmp_path) -> None:
    """Shuffled iteration regression: a frame still needed by the current window but
    loaded long ago sits at the oldest end of the LRU; inserting this window's missing
    frames must not evict it (that raced into KeyError: (episode, frame) at read time).
    """
    dataset, _ = _build_dataset(tmp_path)

    class FakeReader:
        def __init__(self, width: int, value: int) -> None:
            self.width = width
            self.value = value

        def get_frames(self, indices: np.ndarray) -> torch.Tensor:
            return torch.full((len(indices), 3, 4, self.width), self.value, dtype=torch.uint8)

    readers = {"head": FakeReader(width=8, value=1), "right_wrist": FakeReader(width=16, value=2)}
    dataset._get_reader = MethodType(lambda self, episode_idx, camera: readers[camera], dataset)

    cache_size = dataset._composed_frame_cache_size  # 66 = 33 frames x stride 2
    window = np.arange(0, 65, 2, dtype=np.int64)  # the window we will reload
    SingleRightHandRawDataset._load_video(dataset, 0, window)
    filler_a = np.arange(200, 266, 2, dtype=np.int64)  # 33 frames: cache now full
    SingleRightHandRawDataset._load_video(dataset, 0, filler_a)
    filler_b = np.arange(400, 446, 2, dtype=np.int64)  # 23 frames: evicts 23 oldest of `window`
    SingleRightHandRawDataset._load_video(dataset, 0, filler_b)
    assert len(dataset._composed_frame_cache) == cache_size

    # Reloading the first window must refresh its stale survivors, not evict them.
    reloaded = SingleRightHandRawDataset._load_video(dataset, 0, window)
    assert reloaded.shape == (3, 33, 6, 8)


def test_load_video_reads_precomputed_memmap_without_mp4(tmp_path) -> None:
    dataset, _ = _build_dataset(tmp_path, use_precomputed_video=True)

    def fail_get_reader(*args, **kwargs):
        raise AssertionError(f"MP4 reader must not be used: {args}, {kwargs}")

    dataset._get_reader = fail_get_reader
    observation = np.arange(0, 65, 2, dtype=np.int64)

    video = dataset._load_video(1, observation)

    assert video.shape == (3, 33, 6, 8)
    assert video.dtype == torch.uint8
    assert torch.all(video == 1)
    assert list(dataset._video_array_cache) == [1]
