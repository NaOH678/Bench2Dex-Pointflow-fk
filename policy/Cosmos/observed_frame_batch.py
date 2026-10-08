"""Bench2Dex single-observation batch with unchanged zero future placeholders.

Keep the upstream batch contract, normalization and prompt fields. Resize/pad
only the observed image; linear resizing and reflection padding preserve zero
future images exactly. This is only for the single-observation inference path,
not training videos or multi-frame history. Regression tests compare the entire
batch against the supplied WorldAct Bench2Dex implementation.
"""
import json


def build_observed_frame_batch(self, images_rgb, right_state):
    import torch
    from cosmos_framework.inference.robot_policy.bench2dex import compose_rgb, VIEW_DESCRIPTION
    from cosmos_framework.data.generator.action.action_processing import (
        ActionProcessingRecord,
        make_batched_action_processing_fields,
    )
    from cosmos_framework.data.generator.action.domain_utils import get_domain_id
    from cosmos_framework.data.generator.action.json_formatter import (
        ActionPromptJsonFormatter,
    )
    from cosmos_framework.data.generator.action.transforms import (
        build_sequence_plan_from_mode,
        find_closest_target_size,
        reflection_pad_to_target,
    )

    composed = torch.from_numpy(compose_rgb(images_rgb, self.view_width))
    target_frames = self.config.model.native_chunk_size + 1
    _, height, width = composed.shape

    target_w, target_h = find_closest_target_size(height, width, self.config.model.resolution)
    padded = {"video": composed[:, None]}
    reflection_pad_to_target(padded, ["video"], True, target_w, target_h)
    observed = padded["video"]
    video = observed.new_zeros((3, target_frames, *observed.shape[-2:]))
    video[:, 0] = observed[:, 0]
    image_size = padded["image_size"]

    action = torch.zeros((target_frames, self.config.model.max_action_dim), dtype=torch.float32)
    action[0, : self.config.model.native_action_dim] = torch.from_numpy(right_state)
    sequence_plan = build_sequence_plan_from_mode(
        mode="wam",
        video_length=target_frames,
        action_length=target_frames,
        has_text=True,
    )
    prompt_data = {
        "ai_caption": self.config.model.task,
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(self.config.deployment.action_rate_hz),
        "image_size": image_size,
        "mode": "wam",
        "viewpoint": "concat_view",
        "additional_view_description": VIEW_DESCRIPTION,
    }
    formatted = ActionPromptJsonFormatter(caption_key="ai_caption")(prompt_data)["ai_caption"]
    prompt = json.dumps(formatted) if isinstance(formatted, dict) else str(formatted)
    record = ActionProcessingRecord(raw_action_dim=self.config.model.native_action_dim, action_normalizer=None)
    return {
        self.input_video_key: [[video]],
        "action": [[action]],
        **make_batched_action_processing_fields(record, batch_size=1),
        "mode": ["wam"],
        "ai_caption": [prompt],
        "prompt": [prompt],
        "conditioning_fps": [torch.tensor(self.config.deployment.action_rate_hz, dtype=torch.long)],
        "image_size": image_size.unsqueeze(0).to(device=self.device),
        "domain_id": [torch.tensor(get_domain_id(self.config.model.domain_name), dtype=torch.long)],
        "sequence_plan": [sequence_plan],
    }
