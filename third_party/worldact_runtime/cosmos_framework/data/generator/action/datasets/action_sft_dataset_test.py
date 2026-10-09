# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset


class _BlockDataset:
    def __len__(self) -> int:
        return 13

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return [(0, 10), (10, 3)]


def test_iterable_shuffle_splits_large_episode_blocks() -> None:
    dataset = ActionIterableShuffleDataset(_BlockDataset(), max_block_size=4)  # type: ignore[arg-type]

    assert dataset._get_shuffle_blocks() == [(0, 4), (4, 4), (8, 2), (10, 3)]
