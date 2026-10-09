"""Explicit-window dual-hand simulation samples with causal selected anchors."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_processing import ActionAffineNormalization
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionIterableShuffleDataset
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.pointflow_window import read_frame


class SimPointFKSFTDataset(Dataset):
    def __init__(
        self,
        bundle,
        selection_root,
        split,
        tokenizer_config,
        cfg_dropout_rate,
        max_action_dim,
        *,
        cached_video_only=False,
    ):
        self.cached_video_only = cached_video_only
        self.root = Path(bundle)
        self.cache = self.root / "datasets/bench2dex-task21-cosmos-cache"
        self.selected = Path(selection_root)
        selection = json.loads((self.selected / "manifest.json").read_text())
        if selection["schema"] != "sim_pointfk_handguided1024_v1":
            raise ValueError("Unsupported point selection cache")
        self.rows = [r for r in selection["windows"] if r["split"] == ("validation" if split == "val" else split)]
        if not self.rows:
            raise ValueError(f"Empty split {split}")
        self.latent_root = self.cache / "vae_window_latents"
        manifest = json.loads((self.latent_root / "window_manifest.json").read_text())
        contract = manifest["contract"]
        if (contract["schema"], contract["fps"], contract["chunk_length"], contract["storage"]) != (
            "sim_pointfk_explicit_window_latents_v1",
            20,
            32,
            "bfloat16_bits_in_uint16",
        ):
            raise ValueError("Wrong latent contract")
        index_hash = hashlib.sha256((self.cache / "window_index.json").read_bytes()).hexdigest()
        if contract["window_index_sha256"] != index_hash or selection["window_index_sha256"] != index_hash:
            raise ValueError("Selection, latent and dataset window indices differ")
        self.latents = {r["name"]: r for r in manifest["episodes"]}
        self.offsets = {
            name: {start: i for i, start in enumerate(r["start_frames"])} for name, r in self.latents.items()
        }
        norm = json.loads((self.root / "normalization_train_only.json").read_text())
        if not norm["source"].startswith("training episodes only"):
            raise ValueError("Action statistics include validation")
        self.joint_names = norm["joint_names"]
        low, high = [np.asarray(norm["action"][k], np.float32) for k in ("q01", "q99")]
        self.normalizer = ActionAffineNormalization(
            torch.from_numpy((low + high) / 2), torch.from_numpy(np.maximum((high - low) / 2, 0.05))
        )
        self.transform = ActionTransformPipeline(
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            max_action_dim=max_action_dim,
            video_temporal_downsample=4,
            format_prompt_as_json=True,
        )

    def __len__(self):
        return len(self.rows)

    def get_shuffle_blocks(self):
        result = []
        for i, row in enumerate(self.rows):
            if i == 0 or row["episode"] != self.rows[i - 1]["episode"]:
                result.append([i, 1])
            else:
                result[-1][1] += 1
        return [tuple(r) for r in result]

    def __getitem__(self, index):
        row = self.rows[index]
        saved = torch.load(self.selected / row["path"], map_location="cpu", weights_only=False)
        ids = np.asarray(row["frame_ids"])
        name, start = row["episode"], row["start_frame"]
        np.testing.assert_array_equal(saved["raw_frame_ids"], ids)
        if saved["joint_names"] != self.joint_names:
            raise ValueError("Action joint order changed")
        offset = self.offsets[name][start]
        latent_record = self.latents[name]
        np.testing.assert_array_equal(latent_record["frame_ids"][offset], ids)
        rgb_ids = ids[:1] if self.cached_video_only else ids
        frames = np.stack([read_frame(self.cache / "video_frames" / f"{name}.npy", int(i)) for i in rgb_ids])
        bits = read_frame(self.latent_root / latent_record["path"], offset)
        if bits.dtype != np.uint16 or bits.shape != (48, 9, 40, 40):
            raise ValueError("Wrong latent dtype/shape")
        sample = dict(
            video=torch.from_numpy(frames).permute(1, 0, 2, 3).contiguous(),
            action=saved["action"],
            ai_caption=saved["ai_caption"],
            conditioning_fps=torch.tensor(20),
            mode="wam",
            domain_id=torch.tensor(28),
            viewpoint="concat_view",
            additional_view_description="Head on top; left wrist bottom-left; right wrist bottom-right.",
            pointflow=saved["pointflow"],
            fk=saved["fk"],
            episode_name=name,
            raw_frame_ids=ids,
            vae_latent_cache=torch.from_numpy(bits).view(torch.bfloat16).unsqueeze(0),
        )
        if self.cached_video_only:
            # RGB contains only the anchor; targets and full temporal extent live in the cache.
            sample["cached_video_num_frames"] = len(ids)
        result = self.transform(sample, "480", action_normalizer=self.normalizer)
        if result["image_size"].tolist() != latent_record["image_size"]:
            raise ValueError("Spatial transform differs from latent cache")
        return result


def get_sim_pointfk_sft_dataset(
    *,
    bundle,
    selection_root,
    split="train",
    tokenizer_config=None,
    cfg_dropout_rate=0.1,
    max_action_dim=64,
    iterable_shuffle=True,
    use_image_augmentation=False,
    cached_video_only=True,
):
    if use_image_augmentation:
        raise ValueError("Image augmentation is incompatible with fixed VAE latents")
    ds = SimPointFKSFTDataset(
        bundle,
        selection_root,
        split,
        tokenizer_config,
        cfg_dropout_rate,
        max_action_dim,
        # Eval callbacks reconstruct even train cases with iterable_shuffle=False.
        cached_video_only=cached_video_only and split == "train" and iterable_shuffle,
    )
    return ActionIterableShuffleDataset(ds, seed=42, max_block_size=4) if iterable_shuffle else ds
