"""One-case checkpoint evaluation, followed by 3D rendering and process exit."""

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from cosmos_framework.callbacks.pointflow_eval import PointFlowEvalCallback
from cosmos_framework.callbacks.pointflow_eval_cases import evaluation_rng
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn
from cosmos_framework.utils import ema
from cosmos_framework.utils.lazy_config import instantiate


def render_3d(point_xyz, fk_xyz, output, fps=15):
    """Two fixed, equal-scale views; no fitted translation/scale or GT masking."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from cosmos_framework.callbacks.fk_visualize import EDGES

    point_xyz, fk_xyz = np.asarray(point_xyz), np.asarray(fk_xyz)
    if point_xyz.ndim != 3 or fk_xyz.ndim != 3 or point_xyz.shape[0] != fk_xyz.shape[0]:
        raise ValueError("Expected matching [time, points, 3] trajectories")
    if not np.isfinite(point_xyz).all() or not np.isfinite(fk_xyz).all():
        raise ValueError("Non-finite predicted coordinates")
    joined = np.concatenate([point_xyz.reshape(-1, 3), fk_xyz.reshape(-1, 3)])
    lower, upper = joined.min(0), joined.max(0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) * 0.55, 0.02)
    fig = plt.figure(figsize=(12, 6), dpi=120)
    axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    with writer.saving(fig, str(output), dpi=120):
        for t in range(len(point_xyz)):
            for ax, azimuth in zip(axes, (-65, 25), strict=True):
                ax.clear()
                ax.scatter(*point_xyz[t].T, s=7, c="#b34bd4", alpha=0.8, label="Predicted PointFlow")
                ax.scatter(*fk_xyz[t].T, s=18, c="#008fca", depthshade=False, label="Predicted FK")
                for a, b in EDGES:
                    ax.plot(*fk_xyz[t, [a, b]].T, color="#008fca", linewidth=2)
                for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, strict=True):
                    setter(c - radius, c + radius)
                ax.set_box_aspect((1, 1, 1))
                ax.set(xlabel="Camera X (m)", ylabel="Camera Y (m)", zlabel="Camera Z (m)")
                ax.view_init(elev=20, azim=azimuth)
                ax.legend(loc="upper left", fontsize=8)
            fig.suptitle(f"Joint prediction | frame {t:02d} | {t / fps:.2f} s\n"
                         "Fixed camera coordinates; approximate PF/FK registration; frame 0 is observed")
            writer.grab_frame()
    plt.close(fig)


def load_case(config, source, case_id):
    rows = json.loads((source / "pointflow_eval/fixed_cases_stages_4windows.json").read_text())
    identity = next(row for row in rows if row["case_id"] == case_id)
    loader = getattr(config, "dataloader_" + identity["split"])
    cfg = OmegaConf.to_container(loader, resolve=True) if OmegaConf.is_config(loader) else copy.deepcopy(loader)
    datasets = cfg["dataloader"]["datasets"]
    if len(datasets) != 1:
        raise ValueError("Single-case replay requires one dataset")
    dataset_cfg = copy.deepcopy(next(iter(datasets.values()))["dataset"])
    dataset_cfg.update(iterable_shuffle=False, cfg_dropout_rate=0.0, use_image_augmentation=False)
    with evaluation_rng(identity["seed"]):
        dataset = instantiate(dataset_cfg)
        sample = dataset[identity["index"]]
    pf = sample["pointflow"]
    if str(pf["metadata"]["episode"]) != identity["episode"]:
        raise ValueError("Replay episode differs from source")
    np.testing.assert_array_equal(pf["metadata"]["raw_frame_ids"], identity["raw_frame_ids"])
    np.testing.assert_array_equal(pf["inputs"]["point_ids"], identity["point_ids"])
    pack = {k: v for k, v in cfg.items() if k not in ("_target_", "dataloader")}
    pack.update(max_samples_per_batch=1, max_sequence_length=None)
    inner = torch.utils.data.DataLoader([sample], batch_size=1, num_workers=0, collate_fn=custom_collate_fn)
    batch = next(iter(PackingDataLoader(dataloader=inner, **pack)))
    return identity, batch


class SingleCase3DEval(PointFlowEvalCallback):
    @torch.no_grad()
    def on_validation_start(self, model, dataloader, iteration=0):
        from cosmos_framework.callbacks.joint_pointflow_fk_eval import run_joint_eval

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError("This export command requires exactly one GPU")
        source = Path(os.environ["POINTFLOW_3D_SOURCE_JOB"])
        case_id = os.environ.get("POINTFLOW_3D_CASE", "val_01")
        identity, batch = load_case(self.config, source, case_id)
        self._fixed = ([identity], [batch])
        self.sampling_steps = 4
        model.eval()
        with ema.ema_scope(model, enabled=model.config.ema.enabled):
            run_joint_eval(self, model, 10000)
        root = Path(self.config.job.path_local)
        case_dir = root / "pointflow_eval/step_0010000" / (case_id + "_joint")
        with np.load(case_dir / "prediction.npz", allow_pickle=False) as pf:
            point_xyz = pf["xyz0"][None] + pf["flow"]
            frames = pf["raw_frame_ids"].copy()
        with np.load(root / "fk_eval/step_0010000" / case_id / "prediction.npz", allow_pickle=False) as fk:
            np.testing.assert_array_equal(frames, fk["frame_ids"])
            pred = fk["prediction"]
            fk_xyz = fk["anchor"][None] + np.concatenate([np.zeros_like(pred[:1]), pred])
        output = Path(os.environ["OUTPUT_ROOT"]) / f"{case_id}_joint_point_fk_3d.mp4"
        render_3d(point_xyz, fk_xyz, output)
        np.savez_compressed(output.with_suffix(".npz"), point_xyz=point_xyz, fk_xyz=fk_xyz, raw_frame_ids=frames)
        output.with_suffix(".json").write_text(json.dumps(dict(
            source_job=str(source), checkpoint="iter_000010000", case=identity,
            output_job=str(root), coordinates="camera xyz metres, no fitted alignment",
            note="New joint sample; not the original legacy prediction. Frame 0 is conditioned.",
        ), indent=2) + "\n")
        print(f"3D_EXPORT_COMPLETE: {output}", flush=True)
        # This callback runs at startup validation, before any optimizer step.
        raise SystemExit(0)
