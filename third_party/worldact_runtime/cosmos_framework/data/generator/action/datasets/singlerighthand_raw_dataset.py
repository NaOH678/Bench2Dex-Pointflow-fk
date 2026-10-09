# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action-policy adapter for the single-right-hand raw episode format.

The raw LMDB files live on GPFS, where LMDB mmap is not supported. A small
one-time preprocessing step extracts only the aligned state/action arrays into
NPZ sidecars; videos continue to be decoded from the original episode MP4s.
"""

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
import torchvision.transforms.functional as transforms_F
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_spec import ActionSpec, Joint, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.domain_utils import get_domain_id


@dataclass(frozen=True)
class _Episode:
    name: str
    num_frames: int
    source_fps: float


class _TorchCodecFrameReader:
    """Batch-decode CPU frames while the dataset cache handles window overlap."""

    def __init__(self, path: Path) -> None:
        from torchcodec.decoders import VideoDecoder

        self._decoder: Any = VideoDecoder(str(path), device="cpu", num_ffmpeg_threads=1)

    def get_frames(self, indices: np.ndarray) -> torch.Tensor:
        return self._decoder.get_frames_at(indices.tolist()).data

    def close(self) -> None:
        self._decoder = None


class _OpenCVFrameReader:
    """Explicit CPU decoding option for environments without TorchCodec libraries."""

    def __init__(self, path):
        import cv2

        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")

    def get_frames(self, indices):
        import cv2

        frames = []
        for index in indices:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = self._capture.read()
            if not ok:
                raise ValueError(f"Cannot decode video frame {index}")
            frames.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        return torch.stack(frames)

    def close(self):
        self._capture.release()


class SingleRightHandRawDataset(Dataset):
    """Read 30-fps single-right-hand episodes as official 15-fps policy windows."""

    EMBODIMENT_TYPE = "singlerighthand"

    def __init__(
        self,
        *,
        root: str,
        cache_root: str,
        fps: float = 15.0,
        chunk_length: int = 32,
        split: str = "train",
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        action_stats_path: str | None = None,
        action_stats_sha256: str | None = None,
        sample_stride: int = 1,
        mode: str = "wam",
        use_state: bool = True,
        viewpoint: str = "concat_view",
        task_text: str | None = None,
        use_image_augmentation: bool = False,
        use_precomputed_video: bool | str = False,
        video_cache_resolution: str | int | None = None,
        frame_cache_size: int | None = None,
        video_decoder: str = "torchcodec",
        episode_allowlist: str | None = None,
        fk_root: str | None = None,
        fk_steps_per_token: int = 4,
        fk_camera_profile: str = "legacy",
        pointflow_manifest: str | None = None,
        pointflow_max_points: int = 8192,
        pointflow_voxel_size: float = 0.02,
        pointflow_seed: int = 0,
        pointflow_select_motion_fraction: float = 0.0,
        pointflow_select_top_n: int = 0,
        pointflow_min_voxel_members: int = 0,
        pointflow_supervise_cluster_n: int = 0,
        pointflow_select_regions=(),
        pointflow_select_region_quotas=(),
        pointflow_select_min_valid_steps: int = 0,
        pointflow_select_phantom_guard: bool = False,
        pointflow_window_cache_root: str | None = None,
        video_temporal_downsample: int = 4,
        vae_latent_root: str | None = None,
        vae_window_latent_root: str | None = None,
    ) -> None:
        super().__init__()
        if video_decoder not in {"torchcodec", "opencv"}:
            raise ValueError("video_decoder must be torchcodec or opencv")
        self._video_decoder = video_decoder
        self._vae_latent_root = Path(vae_latent_root) if vae_latent_root else None
        self._root = Path(root)
        self._cache_root = Path(cache_root)
        self._fps = float(fps)
        self._chunk_length = int(chunk_length)
        self._video_temporal_downsample = int(video_temporal_downsample)
        self._sample_stride = int(sample_stride)
        if self._sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {self._sample_stride}")
        self._mode = mode
        self._use_state = bool(use_state)
        self._viewpoint = viewpoint
        self._domain_id = get_domain_id(self.EMBODIMENT_TYPE)
        self._use_image_augmentation = use_image_augmentation
        self._image_augmentor: T.Compose | None = None
        if isinstance(use_precomputed_video, str):
            use_precomputed_video = use_precomputed_video.lower() in {"1", "true", "yes", "on"}
        self._use_precomputed_video = bool(use_precomputed_video)
        if self._use_precomputed_video and self._use_image_augmentation:
            raise ValueError("Precomputed video frames cannot be combined with online image augmentation")

        manifest_path = self._cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run tools/prepare_singlerighthand_raw.py before training."
            )
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("schema_version", 0)) != 2:
            raise ValueError(
                f"Unsupported cache schema in {manifest_path}; rerun tools/prepare_singlerighthand_raw.py --overwrite"
            )
        self._arm_action_space = str(manifest.get("arm_action_space", "eef")).lower().strip()
        if self._arm_action_space not in {"eef", "joint"}:
            raise ValueError(
                f"Unsupported arm_action_space={self._arm_action_space!r} in {manifest_path}; expected 'eef' or 'joint'"
            )
        self._task_text = task_text or str(manifest.get("task_text", "make a sandwich"))

        all_episodes = [
            _Episode(
                name=str(row["name"]),
                num_frames=int(row["num_frames"]),
                source_fps=float(row["source_fps"]),
            )
            for row in manifest["episodes"]
        ]
        # Restrict to a named subset before the split: the small-sample workflow
        # trains on ten episodes, and filtering afterwards would change which
        # ones the seed assigns to validation.  Unknown names are an error, not a
        # quiet no-op, because a typo would otherwise shrink the training set.
        if episode_allowlist:
            allow_path = Path(episode_allowlist)
            names = {line.strip() for line in allow_path.read_text().splitlines() if line.strip()}
            if not names:
                raise ValueError(f"Episode allowlist is empty: {allow_path}")
            missing = sorted(names - {episode.name for episode in all_episodes})
            if missing:
                raise ValueError(f"Episode allowlist contains unknown episodes: {missing[:3]}")
            all_episodes = [episode for episode in all_episodes if episode.name in names]
        if not all_episodes:
            raise ValueError(f"No episodes listed in {manifest_path}")

        self._video_cache_rows: dict[str, dict[str, Any]] = {}
        if self._use_precomputed_video:
            video_manifest_path = self._cache_root / "video_manifest.json"
            if not video_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing {video_manifest_path}. Run tools/prepare_singlerighthand_video_cache.py first."
                )
            video_manifest = json.loads(video_manifest_path.read_text())
            if int(video_manifest.get("schema_version", 0)) != 1:
                raise ValueError(f"Unsupported video cache schema in {video_manifest_path}")
            if video_cache_resolution is not None and str(video_manifest.get("resolution")) != str(
                video_cache_resolution
            ):
                raise ValueError(
                    f"Video cache resolution={video_manifest.get('resolution')} does not match "
                    f"training resolution={video_cache_resolution}"
                )
            self._video_cache_rows = {str(row["name"]): row for row in video_manifest["episodes"]}
            missing = [episode.name for episode in all_episodes if episode.name not in self._video_cache_rows]
            if missing:
                raise ValueError(f"Video cache is missing {len(missing)} episodes; first missing episode: {missing[0]}")

        split_key = split.lower().strip()
        if split_key not in {"train", "val", "full"}:
            raise ValueError(f"Unsupported split={split!r}; expected train, val, or full")
        num_val = int(round(len(all_episodes) * split_val_ratio))
        order = torch.randperm(len(all_episodes), generator=torch.Generator().manual_seed(split_seed)).tolist()
        selected = order[:num_val] if split_key == "val" else order[num_val:] if split_key == "train" else order
        self._episodes = [all_episodes[idx] for idx in selected]
        self.action_normalizer = None
        if action_stats_path:
            from cosmos_framework.data.generator.action.real_normalization import RealActionNormalizer

            self.action_normalizer = RealActionNormalizer(action_stats_path, expected_sha256=action_stats_sha256)
            self.action_normalizer.validate_dataset(
                cache_root=self._cache_root,
                all_episodes=all_episodes,
                order=order,
                num_val=num_val,
                split_seed=split_seed,
                split_val_ratio=split_val_ratio,
                action_names=self.action_names,
                arm_action_space=self.arm_action_space,
            )
        elif action_stats_sha256:
            raise ValueError("action_stats_sha256 requires action_stats_path")

        # Point-cloud labels are optional and keyed by episode: absent means "this
        # run does not train the PointFlow modality", never a missing label.
        self._pointflow_source = None
        # An empty environment override disables PointFlow (e.g. the WAM baseline).
        # Path("") would otherwise resolve to the working directory.
        if pointflow_manifest:
            from cosmos_framework.data.generator.action.pointflow_source import PointFlowSource
            from cosmos_framework.data.pointflow_window import PointFlowTiming

            if use_image_augmentation:
                raise ValueError(
                    "PointFlow requires recorded spatial transforms; random image augmentation is unsupported"
                )
            self._pointflow_source = PointFlowSource(
                pointflow_manifest,
                timing=PointFlowTiming(self._fps, self._chunk_length, video_temporal_downsample),
                max_points=pointflow_max_points,
                voxel_size=pointflow_voxel_size,
                seed=pointflow_seed,
                # GT-ranked restriction to the most-moving voxels; a mechanism
                # diagnostic, not a deployable conditioning rule (see prepare_window).
                select_motion_fraction=pointflow_select_motion_fraction,
                select_top_n=pointflow_select_top_n,
                min_voxel_members=pointflow_min_voxel_members,
                supervise_cluster_n=pointflow_supervise_cluster_n,
                select_regions=pointflow_select_regions,
                select_region_quotas=pointflow_select_region_quotas,
                select_min_valid_steps=pointflow_select_min_valid_steps,
                select_phantom_guard=pointflow_select_phantom_guard,
                window_cache_root=pointflow_window_cache_root,
            )
            if self._pointflow_source.window_cache_root is not None:
                # Same contract as the VAE window-latent cache below: the cache is
                # keyed by window offset, so anything that changes the enumeration
                # invalidates it.  Refuse rather than read the wrong window.
                from cosmos_framework.data.pointflow_window_cache import ENUMERATION_KEYS, validate_manifest

                validate_manifest(
                    self._pointflow_source.window_cache_root,
                    {
                        "fps": self._fps,
                        "chunk_length": self._chunk_length,
                        "sample_stride": self._sample_stride,
                    },
                    ENUMERATION_KEYS,
                )
            missing = [e.name for e in self._episodes if e.name not in self._pointflow_source.entries]
            if missing:
                raise ValueError(
                    f"PointFlow manifest must explicitly list selected episodes, including unlabeled: {missing}"
                )

        # FK labels are optional and keyed by episode, like the point-cloud source:
        # absent means "this run does not train the FK modality", never a missing label.
        # Truthy, not ``is not None``: ``${oc.env:FK_ANNOTATION_ROOT,}`` resolves to
        # the EMPTY STRING when unset, and ``Path('')`` is the current directory --
        # so an ``is not None`` check would treat "modality disabled" as "annotations
        # live in the CWD" and fail much later with a confusing missing-file error.
        self._fk_source = None
        if fk_root:
            from cosmos_framework.data.fk_window import FKTiming
            from cosmos_framework.data.generator.action.fk_source import FKSource

            timing = FKTiming(fps=self._fps, steps=self._chunk_length, steps_per_token=int(fk_steps_per_token))
            self._fk_source = FKSource(fk_root, timing=timing, camera_profile=fk_camera_profile)
            missing = [e.name for e in self._episodes if not self._fk_source.path_for(e.name).is_file()]
            if missing:
                raise ValueError(f"FK annotations are missing for selected episodes: {missing[:3]}")

        # Precomputed per-window VAE latents.  Without this the frozen Wan2.2 VAE
        # re-encodes every pixel frame of every window on every step; with it the
        # dataset hands the model the latent the encode would have produced.
        # Only the WINDOW cache is accepted -- see the note in ``_read_window_latent``
        # for why the whole-episode cache is deliberately not an option here.
        self._vae_window_latent_root = Path(vae_window_latent_root) if vae_window_latent_root else None
        self._vae_window_rows: dict[str, dict[str, Any]] = {}
        if self._vae_window_latent_root is not None:
            window_manifest_path = self._vae_window_latent_root / "window_manifest.json"
            if not window_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing {window_manifest_path}. Run tools/cache_window_vae_latents.py before training."
                )
            window_manifest = json.loads(window_manifest_path.read_text())
            if int(window_manifest.get("schema_version", 0)) != 1:
                raise ValueError(f"Unsupported window latent cache schema in {window_manifest_path}")
            # The cache is keyed by window offset, so anything that changes the
            # window enumeration invalidates it. Refuse rather than read the wrong
            # window's latent.
            for field, current in (
                ("fps", self._fps),
                ("chunk_length", self._chunk_length),
                ("sample_stride", self._sample_stride),
            ):
                if float(window_manifest[field]) != float(current):
                    raise ValueError(
                        f"{window_manifest_path}: {field}={window_manifest[field]} but this dataset uses "
                        f"{current}; regenerate the window latent cache"
                    )
            self._vae_window_rows = {str(row["name"]): row for row in window_manifest["episodes"]}
            missing = [e.name for e in self._episodes if e.name not in self._vae_window_rows]
            if missing:
                raise ValueError(f"Window latent cache has no entry for selected episodes: {missing[:3]}")

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
        self._reader_cache: OrderedDict[tuple[int, str], _TorchCodecFrameReader | _OpenCVFrameReader] = OrderedDict()
        self._reader_cache_size = 2
        self._video_array_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._video_array_cache_size = 2
        self._frame_cache_size = frame_cache_size or (self._chunk_length + 1) * max(self._source_strides)
        self._composed_frame_cache: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()
        self._composed_frame_cache_size = self._frame_cache_size

    @property
    def action_dim(self) -> int:
        return 27

    @property
    def action_spec(self) -> ActionSpec:
        if self._arm_action_space == "joint":
            return build_action_spec(
                Joint(n=7, label="arm", prefix="right"),
                Joint(n=20, label="hand", prefix="right"),
            )
        return build_action_spec(
            Pos(prefix="right"), Rot("quat_xyzw", prefix="right"), Joint(n=20, label="hand", prefix="right")
        )

    @property
    def arm_action_space(self) -> str:
        return self._arm_action_space

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
        with np.load(self._cache_root / "episodes" / f"{episode.name}.npz") as data:
            state = np.asarray(data["state"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
        if state.shape != (episode.num_frames, self.action_dim) or action.shape != state.shape:
            raise ValueError(
                f"Episode {episode.name}: expected state/action {(episode.num_frames, self.action_dim)}, "
                f"got {state.shape}/{action.shape}"
            )
        if self.action_normalizer is not None:
            state = self.action_normalizer.normalize(state)
            action = self.action_normalizer.normalize(action)
        cached = (state, action)
        self._array_cache[episode_idx] = cached
        while len(self._array_cache) > self._array_cache_size:
            self._array_cache.popitem(last=False)
        return cached

    def _get_reader(self, episode_idx: int, camera: str) -> _TorchCodecFrameReader | _OpenCVFrameReader:
        key = (episode_idx, camera)
        reader = self._reader_cache.get(key)
        if reader is not None:
            self._reader_cache.move_to_end(key)
            return reader
        path = self._root / self._episodes[episode_idx].name / "videos" / f"{camera}.mp4"
        reader = (_TorchCodecFrameReader if self._video_decoder == "torchcodec" else _OpenCVFrameReader)(path)
        self._reader_cache[key] = reader
        while len(self._reader_cache) > self._reader_cache_size:
            _, old_reader = self._reader_cache.popitem(last=False)
            old_reader.close()
        return reader

    @staticmethod
    def _close_memmap(array: np.ndarray) -> None:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def _get_precomputed_video(self, episode_idx: int) -> np.ndarray:
        cached = self._video_array_cache.get(episode_idx)
        if cached is not None:
            self._video_array_cache.move_to_end(episode_idx)
            return cached

        episode = self._episodes[episode_idx]
        row = self._video_cache_rows[episode.name]
        path = self._cache_root / str(row["path"])
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(int(value) for value in row["shape"])
        if cached.dtype != np.uint8 or cached.shape != expected_shape:
            self._close_memmap(cached)
            raise ValueError(
                f"Episode {episode.name}: expected uint8 video cache {expected_shape}, got {cached.dtype} {cached.shape}"
            )
        if expected_shape[0] != episode.num_frames or expected_shape[1] != 3:
            self._close_memmap(cached)
            raise ValueError(
                f"Episode {episode.name}: video cache must have shape [num_frames,3,H,W], got {expected_shape}"
            )

        self._video_array_cache[episode_idx] = cached
        while len(self._video_array_cache) > self._video_array_cache_size:
            _, old_array = self._video_array_cache.popitem(last=False)
            self._close_memmap(old_array)
        return cached

    @staticmethod
    def _compose_views(head: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        """Stack aspect-preserved wrist and head views at a shared width."""
        target_width = min(head.shape[-1], wrist.shape[-1])

        def resize_to_width(video: torch.Tensor) -> torch.Tensor:
            height, width = video.shape[-2:]
            target_height = max(1, round(height * target_width / width))
            if (height, width) == (target_height, target_width):
                return video
            return transforms_F.resize(
                video,
                [target_height, target_width],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )

        wrist = resize_to_width(wrist)
        head = resize_to_width(head)
        return torch.cat([wrist, head], dim=-2)  # [T,C,H_wrist+H_head,W]

    def _load_video(self, episode_idx: int, observation_indices: np.ndarray) -> torch.Tensor:
        # Refresh recency of cache hits BEFORE inserting: the eviction below pops
        # the oldest entries, and a needed-but-stale frame (loaded for an earlier
        # window under shuffled iteration) would otherwise be evicted by this very
        # call and surface as a KeyError in the read loop.
        for frame_idx in observation_indices:
            key = (episode_idx, int(frame_idx))
            if key in self._composed_frame_cache:
                self._composed_frame_cache.move_to_end(key)
        missing_indices = [
            int(frame_idx)
            for frame_idx in observation_indices
            if (episode_idx, int(frame_idx)) not in self._composed_frame_cache
        ]
        if missing_indices:
            missing = np.asarray(missing_indices, dtype=np.int64)
            if self._use_precomputed_video:
                array = self._get_precomputed_video(episode_idx)
                composed = torch.from_numpy(np.array(array[missing], copy=True))
            else:
                head = self._get_reader(episode_idx, "head").get_frames(missing)
                wrist = self._get_reader(episode_idx, "right_wrist").get_frames(missing)
                composed = self._compose_views(head, wrist)
            for frame_idx, frame in zip(missing_indices, composed, strict=True):
                self._composed_frame_cache[(episode_idx, frame_idx)] = frame
            while len(self._composed_frame_cache) > self._composed_frame_cache_size:
                self._composed_frame_cache.popitem(last=False)

        frames = []
        for frame_idx_value in observation_indices:
            key = (episode_idx, int(frame_idx_value))
            frame = self._composed_frame_cache[key]
            self._composed_frame_cache.move_to_end(key)
            frames.append(frame)
        video = torch.stack(frames)
        if self._use_image_augmentation:
            if self._image_augmentor is None:
                height, width = video.shape[-2:]
                self._image_augmentor = T.Compose(
                    [
                        T.RandomCrop((int(height * 0.95), int(width * 0.95))),
                        T.Resize((height, width), antialias=True),
                        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                    ]
                )
            video = self._image_augmentor(video)
        return video.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W], uint8

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_idx, window_offset = self._resolve_index(idx)
        observation_indices, action_indices = self._window_indices(episode_idx, window_offset)
        state, action = self._load_arrays(episode_idx)
        raw_action = torch.from_numpy(action[action_indices].copy()).float()
        if self._use_state:
            initial_state = torch.from_numpy(state[observation_indices[0]].copy()).float()
            raw_action = torch.cat([initial_state.unsqueeze(0), raw_action], dim=0)

        mode = random.choice(("forward_dynamics", "inverse_dynamics", "wam")) if self._mode == "joint" else self._mode
        sample = {
            "ai_caption": self._task_text,
            "video": self._load_video(episode_idx, observation_indices),
            "action": raw_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            "additional_view_description": (
                "The upper view is from the right wrist-mounted camera. "
                "The lower view is from the head-mounted third-person camera."
            ),
        }
        if self._pointflow_source is not None:
            episode = self._episodes[episode_idx]
            sample["episode_name"] = episode.name
            sample["raw_frame_ids"] = observation_indices.copy()
            sample["action_frame_ids"] = action_indices.copy()
            sample["timestamps_sec"] = observation_indices / episode.source_fps
            canvas_size = (sample["video"].shape[-1], sample["video"].shape[-2])
            if self._use_precomputed_video and episode.name in self._video_cache_rows:
                target = self._video_cache_rows[episode.name].get("image_size")
                if target and len(target) >= 2:
                    canvas_size = (int(target[1]), int(target[0]))
            sample["pointflow"] = self._pointflow_source.load(
                episode.name,
                observation_indices,
                episode.source_fps,
                canvas_size,
            )
        if self._fk_source is not None:
            # The window is the dataset's own observation_indices, not a recomputed
            # one: the labels must describe the same source frames the video does,
            # or the mRoPE time axis drifts with no shape error to catch it.
            sample["fk"] = self._fk_source.load(self._episodes[episode_idx].name, observation_indices)
            if os.environ.get("FK_PROJECT_ANCHORS", "false").lower() == "true":
                from cosmos_framework.data.generator.action.fk_source import attach_fk_projection

                attach_fk_projection(sample["fk"], sample.get("pointflow"))
        if self._vae_window_latent_root is not None:
            # This window's own encode.  The model prefers it over running the VAE,
            # so the video frames above are still decoded -- they feed the eval
            # renderer and the frame cache -- but they no longer reach the encoder.
            sample["vae_latent_cache"] = self._read_window_latent(self._episodes[episode_idx].name, window_offset)
        elif self._vae_latent_root is not None:
            latent_path = self._vae_latent_root / f"{self._episodes[episode_idx].name}.pt"
            if latent_path.is_file():
                cached = torch.load(latent_path, map_location="cpu", weights_only=True)
                latent = cached.get("latent", cached) if isinstance(cached, dict) else cached
                # Legacy path: a whole-episode encode at source FPS, resampled onto
                # the window's 15 Hz lattice. Measured against a fresh online encode
                # this differs by about one frame of window shift, and the
                # conditioning latent (index 0) becomes a source-frame block rather
                # than the current frame. Keep it only for A/B; prefer the window
                # cache above.
                if isinstance(latent, torch.Tensor) and latent.ndim == 5:
                    source_ids = observation_indices[:: self._video_temporal_downsample]
                    latent_ids = np.floor(source_ids / 4.0).astype(np.int64)
                    latent_ids = np.clip(latent_ids, 0, latent.shape[2] - 1)
                    sample["vae_latent_cache"] = latent.index_select(2, torch.from_numpy(latent_ids))
                else:
                    sample["vae_latent_cache"] = latent
        return sample

    def _read_window_latent(self, name: str, window_offset: int):
        """One window's latent from vae_window_latents/<name>.npy.

        Stored as bfloat16 bit patterns in uint16; the manifest's ``storage`` field
        is asserted so a layout change cannot be read back as numbers silently.
        """
        row = self._vae_window_rows[name]
        with (self._vae_window_latent_root / row["path"]).open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            if tuple(shape[1:]) != tuple(row["shape"]) or dtype != np.uint16:
                raise ValueError(f"{name}: window latent header {shape} {dtype} does not match the manifest")
            if not 0 <= window_offset < shape[0]:
                raise ValueError(f"{name}: window {window_offset} outside the cache's 0..{shape[0] - 1}")
            header = stream.tell()
            item = int(np.prod(shape[1:])) * dtype.itemsize
            stream.seek(header + window_offset * item)
            raw = np.frombuffer(stream.read(item), dtype=np.uint16).reshape(shape[1:])
        # np.frombuffer is read-only and torch needs a writable buffer for a view.
        return torch.from_numpy(raw.copy()).view(torch.bfloat16).unsqueeze(0)

    def __del__(self) -> None:
        for reader in getattr(self, "_reader_cache", {}).values():
            reader.close()
        for array in getattr(self, "_video_array_cache", {}).values():
            self._close_memmap(array)


__all__ = ["SingleRightHandRawDataset"]
