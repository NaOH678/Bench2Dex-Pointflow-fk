# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cache-only action-policy adapter for dual-arm, dual-hand joint data."""

from __future__ import annotations

import json
import os
import random
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_spec import ActionSpec, Joint, build_action_spec
from cosmos_framework.data.generator.action.domain_utils import get_domain_id


@dataclass(frozen=True)
class _Episode:
    name: str
    num_frames: int
    source_fps: float


class _NpyFrameReader:
    """Read selected TCHW frames with ``pread`` when the filesystem cannot mmap."""

    def __init__(self, path: Path, expected_shape: tuple[int, ...]) -> None:
        with path.open("rb") as file:
            version = np.lib.format.read_magic(file)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(file)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(file)
            else:
                raise ValueError(f"Unsupported NPY version {version} in {path}")
            self._data_offset = file.tell()
        self.shape = tuple(int(value) for value in shape)
        if fortran_order or np.dtype(dtype) != np.dtype(np.uint8) or self.shape != expected_shape:
            raise ValueError(
                f"Expected C-order uint8 video cache {expected_shape} in {path}, "
                f"got fortran_order={fortran_order}, dtype={dtype}, shape={self.shape}"
            )
        self._frame_shape = self.shape[1:]
        self._frame_bytes = int(np.prod(self._frame_shape, dtype=np.int64))
        expected_size = self._data_offset + self.shape[0] * self._frame_bytes
        if path.stat().st_size != expected_size:
            raise ValueError(f"Truncated NPY cache {path}: expected {expected_size} bytes, got {path.stat().st_size}")
        self._fd = os.open(path, os.O_RDONLY)

    def get_frames(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.shape[0]):
            raise IndexError(f"Frame indices outside [0, {self.shape[0]}): {indices}")
        frames = np.empty((len(indices), *self._frame_shape), dtype=np.uint8)
        for output_index, frame_index in enumerate(indices):
            offset = self._data_offset + int(frame_index) * self._frame_bytes
            payload = os.pread(self._fd, self._frame_bytes, offset)
            if len(payload) != self._frame_bytes:
                raise OSError(
                    f"Short read for frame {int(frame_index)}: got {len(payload)} of {self._frame_bytes} bytes"
                )
            frames[output_index] = np.frombuffer(payload, dtype=np.uint8).reshape(self._frame_shape)
        return frames

    def close(self) -> None:
        fd = getattr(self, "_fd", -1)
        if fd >= 0:
            os.close(fd)
            self._fd = -1


class DualHandJointCacheDataset(Dataset):
    """Read 54D dual-hand joint windows from precomputed NPZ/NPY caches."""

    EMBODIMENT_TYPE = "dualhand"
    ACTION_DIM = 54
    STATE_SCHEMA = "cosmos_dualhand_joint_v1"
    VIDEO_SCHEMA = "cosmos_dualhand_joint_video_v1"

    def __init__(
        self,
        *,
        cache_root: str,
        fps: float = 15.0,
        chunk_length: int = 32,
        split: str = "train",
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        sample_stride: int = 1,
        mode: str = "wam",
        use_state: bool = True,
        viewpoint: str = "concat_view",
        task_text: str | None = None,
        video_cache_resolution: str | int | None = None,
        frame_cache_size: int | None = None,
    ) -> None:
        super().__init__()
        self._cache_root = Path(cache_root)
        self._fps = float(fps)
        self._chunk_length = int(chunk_length)
        self._sample_stride = int(sample_stride)
        if self._fps <= 0:
            raise ValueError(f"fps must be positive, got {self._fps}")
        if self._chunk_length < 1:
            raise ValueError(f"chunk_length must be >= 1, got {self._chunk_length}")
        if self._sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {self._sample_stride}")
        self._mode = mode
        self._use_state = bool(use_state)
        self._viewpoint = viewpoint
        self._domain_id = get_domain_id(self.EMBODIMENT_TYPE)

        manifest_path = self._cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != self.STATE_SCHEMA or int(manifest.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported dual-hand state cache schema in {manifest_path}")
        self._arm_action_space = str(manifest.get("arm_action_space", "")).lower().strip()
        if self._arm_action_space != "joint":
            raise ValueError(f"Dual-hand cache {manifest_path} must declare arm_action_space='joint'")
        if int(manifest.get("state_dim", 0)) != self.ACTION_DIM or int(manifest.get("action_dim", 0)) != self.ACTION_DIM:
            raise ValueError(f"Dual-hand state/action dimensions in {manifest_path} must both be {self.ACTION_DIM}")
        units = manifest.get("units", {})
        if units.get("state") != "radian" or units.get("action") != "radian":
            raise ValueError(f"Dual-hand state/action units in {manifest_path} must both be radians")
        self._task_text = task_text or str(manifest.get("task_text", ""))

        all_episodes = [
            _Episode(
                name=str(row["name"]),
                num_frames=int(row["num_frames"]),
                source_fps=float(row["source_fps"]),
            )
            for row in manifest["episodes"]
        ]
        if not all_episodes:
            raise ValueError(f"No episodes listed in {manifest_path}")
        if int(manifest.get("num_episodes", len(all_episodes))) != len(all_episodes):
            raise ValueError(f"num_episodes does not match episode rows in {manifest_path}")

        video_manifest_path = self._cache_root / "video_manifest.json"
        if not video_manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {video_manifest_path}; dual-hand training requires the precomputed video cache"
            )
        video_manifest = json.loads(video_manifest_path.read_text())
        if video_manifest.get("schema") != self.VIDEO_SCHEMA or int(video_manifest.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported dual-hand video cache schema in {video_manifest_path}")
        if video_manifest.get("dtype") != "uint8" or video_manifest.get("layout") != "TCHW":
            raise ValueError(f"Dual-hand video cache in {video_manifest_path} must be uint8 TCHW")
        if video_cache_resolution is not None and str(video_manifest.get("resolution")) != str(
            video_cache_resolution
        ):
            raise ValueError(
                f"Video cache resolution={video_manifest.get('resolution')} does not match "
                f"training resolution={video_cache_resolution}"
            )
        self._video_cache_rows = {str(row["name"]): row for row in video_manifest["episodes"]}
        episode_names = {episode.name for episode in all_episodes}
        if set(self._video_cache_rows) != episode_names:
            raise ValueError("Dual-hand numeric and video manifests list different episodes")

        split_key = split.lower().strip()
        if split_key not in {"train", "val", "full"}:
            raise ValueError(f"Unsupported split={split!r}; expected train, val, or full")
        num_val = int(round(len(all_episodes) * split_val_ratio))
        order = torch.randperm(len(all_episodes), generator=torch.Generator().manual_seed(split_seed)).tolist()
        selected = order[:num_val] if split_key == "val" else order[num_val:] if split_key == "train" else order
        self._episodes = [all_episodes[index] for index in selected]

        self._source_strides: list[int] = []
        self._valid_windows: list[int] = []
        self._cumulative_ends: list[int] = []
        total = 0
        for episode in self._episodes:
            stride_float = episode.source_fps / self._fps
            source_stride = int(round(stride_float))
            if source_stride < 1 or not np.isclose(stride_float, source_stride, atol=1e-6):
                raise ValueError(
                    f"Episode {episode.name}: source_fps={episode.source_fps} is not an integer multiple of fps={self._fps}"
                )
            last_start = episode.num_frames - 1 - self._chunk_length * source_stride
            valid_windows = last_start // self._sample_stride + 1 if last_start >= 0 else 0
            self._source_strides.append(source_stride)
            self._valid_windows.append(valid_windows)
            total += valid_windows
            self._cumulative_ends.append(total)

        self._array_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._array_cache_size = 2
        self._video_reader_cache: OrderedDict[int, _NpyFrameReader] = OrderedDict()
        self._video_reader_cache_size = 2
        default_frame_cache_size = (self._chunk_length + 1) * max(self._source_strides, default=1)
        self._frame_cache_size = frame_cache_size or default_frame_cache_size
        self._frame_cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    @property
    def arm_action_space(self) -> str:
        return self._arm_action_space

    @property
    def action_spec(self) -> ActionSpec:
        return build_action_spec(
            Joint(n=7, label="arm", prefix="left"),
            Joint(n=20, label="hand", prefix="left"),
            Joint(n=7, label="arm", prefix="right"),
            Joint(n=20, label="hand", prefix="right"),
        )

    @property
    def action_names(self) -> list[str]:
        return self.action_spec.names

    def __len__(self) -> int:
        return self._cumulative_ends[-1] if self._cumulative_ends else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        start = 0
        for length in self._valid_windows:
            if length > 0:
                blocks.append((start, length))
            start += length
        return blocks

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        episode_idx = bisect_right(self._cumulative_ends, idx)
        block_start = 0 if episode_idx == 0 else self._cumulative_ends[episode_idx - 1]
        return episode_idx, idx - block_start

    def _window_indices(self, episode_idx: int, window_offset: int) -> tuple[np.ndarray, np.ndarray]:
        source_stride = self._source_strides[episode_idx]
        raw_start = window_offset * self._sample_stride
        observation = raw_start + np.arange(self._chunk_length + 1, dtype=np.int64) * source_stride
        return observation, observation[:-1]

    def _load_arrays(self, episode_idx: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._array_cache.get(episode_idx)
        if cached is not None:
            self._array_cache.move_to_end(episode_idx)
            return cached
        episode = self._episodes[episode_idx]
        with np.load(self._cache_root / "episodes" / f"{episode.name}.npz", allow_pickle=False) as data:
            state = np.asarray(data["state"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
        expected = (episode.num_frames, self.action_dim)
        if state.shape != expected or action.shape != expected:
            raise ValueError(f"Episode {episode.name}: expected state/action {expected}, got {state.shape}/{action.shape}")
        cached = (state, action)
        self._array_cache[episode_idx] = cached
        while len(self._array_cache) > self._array_cache_size:
            self._array_cache.popitem(last=False)
        return cached

    def _get_video_reader(self, episode_idx: int) -> _NpyFrameReader:
        reader = self._video_reader_cache.get(episode_idx)
        if reader is not None:
            self._video_reader_cache.move_to_end(episode_idx)
            return reader
        episode = self._episodes[episode_idx]
        row = self._video_cache_rows[episode.name]
        path = self._cache_root / str(row["path"])
        expected_shape = tuple(int(value) for value in row["shape"])
        if expected_shape[0] != episode.num_frames or expected_shape[1] != 3:
            raise ValueError(
                f"Episode {episode.name}: video cache must have shape [num_frames,3,H,W], got {expected_shape}"
            )
        reader = _NpyFrameReader(path, expected_shape)
        self._video_reader_cache[episode_idx] = reader
        while len(self._video_reader_cache) > self._video_reader_cache_size:
            _, old_reader = self._video_reader_cache.popitem(last=False)
            old_reader.close()
        return reader

    def _load_video(self, episode_idx: int, observation_indices: np.ndarray) -> torch.Tensor:
        missing_indices = [
            int(frame_idx)
            for frame_idx in observation_indices
            if (episode_idx, int(frame_idx)) not in self._frame_cache
        ]
        if missing_indices:
            reader = self._get_video_reader(episode_idx)
            frames = torch.from_numpy(reader.get_frames(np.asarray(missing_indices, dtype=np.int64)))
            for frame_idx, frame in zip(missing_indices, frames, strict=True):
                self._frame_cache[(episode_idx, frame_idx)] = frame
            while len(self._frame_cache) > self._frame_cache_size:
                self._frame_cache.popitem(last=False)

        frames = []
        for frame_idx_value in observation_indices:
            key = (episode_idx, int(frame_idx_value))
            frame = self._frame_cache[key]
            self._frame_cache.move_to_end(key)
            frames.append(frame)
        return torch.stack(frames).permute(1, 0, 2, 3).contiguous()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_idx, window_offset = self._resolve_index(idx)
        observation_indices, action_indices = self._window_indices(episode_idx, window_offset)
        state, action = self._load_arrays(episode_idx)
        raw_action = torch.from_numpy(action[action_indices].copy()).float()
        if self._use_state:
            initial_state = torch.from_numpy(state[observation_indices[0]].copy()).float()
            raw_action = torch.cat([initial_state.unsqueeze(0), raw_action], dim=0)

        mode = random.choice(("forward_dynamics", "inverse_dynamics", "wam")) if self._mode == "joint" else self._mode
        return {
            "ai_caption": self._task_text,
            "video": self._load_video(episode_idx, observation_indices),
            "action": raw_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "additional_view_description": (
                "The upper view is from the head-mounted third-person camera. "
                "The lower-left view is from the left wrist-mounted camera, and the lower-right view is from "
                "the right wrist-mounted camera."
            ),
        }

    def __del__(self) -> None:
        for reader in getattr(self, "_video_reader_cache", {}).values():
            reader.close()


__all__ = ["DualHandJointCacheDataset"]
