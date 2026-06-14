"""Save SAM3 per-INSTANCE text-prompt masks for ScanNet subset frames.

Extends save_text_prompt_masks.py: instead of unioning all object masks per
class, this script keeps SAM3's per-object masklets (cross-frame tracked via
`out_obj_ids` from video propagation) and writes one directory per physical
instance:

    <scene>/raw_data/masks_instance/<class>_<k>/<frame>.png

Conventions match the existing per-class masks/: uint8 PNG, {0, 255},
full resolution, one PNG for every subset frame in every instance dir
(all-zero where the instance is not visible).

Association policy:
- SAM3's video predictor assigns a persistent obj_id to each masklet and
  tracks it across frames (the detector re-detects on every frame and the
  tracker matches detections to existing masklets, suppressing duplicates).
  Instance index k is assigned per class in order of first appearance
  (frame index, then obj_id) and never renumbered.
- Lost-track re-association (safety net on top of SAM3's own matching):
  when a brand-new obj_id first appears at frame t, it is compared against
  instances of the same class that are currently invisible and were last
  seen within REASSOC_MAX_GAP subset frames; if mask IoU with the last seen
  mask is >= REASSOC_IOU_THRESH, the new obj_id is merged into that existing
  instance (greedy, best IoU first). Otherwise it becomes a new instance.
- "Stuff" classes (wall, floor) always produce exactly one instance
  (<class>_0) as the union of all object masks.

Per scene the script also writes:
    masks_instance/_qa/overview.jpg   color-consistent QA strip (~10 frames)
    masks_instance/_qa/stats.json     instance counts, union-IoU vs masks/,
                                      re-association events, wall-clock time
    masks_instance/.complete          marker for resumability
"""

from __future__ import annotations

import argparse
import colorsys
import io
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor

# (prompt text, directory name) — same taxonomy/order as save_text_prompt_masks.py
CLASSES = [
    ("wall", "wall"),
    ("floor", "floor"),
    ("cabinet", "cabinet"),
    ("bed", "bed"),
    ("chair", "chair"),
    ("sofa", "sofa"),
    ("table", "table"),
    ("door", "door"),
    ("window", "window"),
    ("bookshelf", "bookshelf"),
    ("picture", "picture"),
    ("counter", "counter"),
    ("desk", "desk"),
    ("curtain", "curtain"),
    ("refrigerator", "refrigerator"),
    ("shower curtain", "shower_curtain"),
    ("toilet", "toilet"),
    ("sink", "sink"),
    ("bathtub", "bathtub"),
]

STUFF_CLASSES = {"wall", "floor"}

REASSOC_IOU_THRESH = 0.3
REASSOC_MAX_GAP = 20  # subset frames (stride-5, so ~100 raw frames / ~3.3 s)

MIN_INSTANCE_PIXELS = 200  # drop specks below this area in every frame they appear


def collect_frame_names(image_dir: Path) -> list[str]:
    names = sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not names:
        raise RuntimeError(f"No frames found in {image_dir}")
    return names


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


class PackedMask:
    """Bit-packed bool mask to keep 100-frame buffers small."""

    def __init__(self, mask: np.ndarray):
        self.shape = mask.shape
        self.bits = np.packbits(mask)
        self.area = int(mask.sum())

    def unpack(self) -> np.ndarray:
        n = self.shape[0] * self.shape[1]
        return np.unpackbits(self.bits, count=n).reshape(self.shape).astype(bool)


def extract_frame_objects(outputs: dict) -> tuple[list[int], list[np.ndarray]]:
    """Return (obj_ids, full-res bool masks) for one frame's outputs."""
    masks = outputs.get("out_binary_masks")
    ids = outputs.get("out_obj_ids")
    if masks is None or ids is None or len(ids) == 0:
        return [], []
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    out_ids, out_masks = [], []
    for obj_id, m in zip(ids, masks):
        if m.ndim == 3:
            m = m[0]
        m = m.astype(bool)
        if m.sum() < MIN_INSTANCE_PIXELS:
            continue
        out_ids.append(int(obj_id))
        out_masks.append(m)
    return out_ids, out_masks


def propagate(predictor, session_id):
    per_frame = {}
    for resp in predictor.handle_stream_request(
        {"type": "propagate_in_video", "session_id": session_id}
    ):
        ids, masks = extract_frame_objects(resp["outputs"])
        per_frame[resp["frame_index"]] = (ids, [PackedMask(m) for m in masks])
    return per_frame


def save_mask_png(mask_bool: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L").save(path)


def zero_png_bytes(height: int, width: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L").save(
        buf, format="PNG"
    )
    return buf.getvalue()


def associate_instances(
    per_frame: dict[int, tuple[list[int], list[PackedMask]]],
    num_frames: int,
) -> tuple[dict[int, dict[int, PackedMask]], list[dict]]:
    """Map raw obj_ids to stable instance indices k (order of first appearance).

    Returns (instance_masks, events) where instance_masks[k][frame_idx] is the
    instance's mask in that frame, and events logs re-associations.
    """
    obj_to_k: dict[int, int] = {}
    # k -> (last_seen_frame, last_seen_mask)
    last_seen: dict[int, tuple[int, PackedMask]] = {}
    instance_masks: dict[int, dict[int, PackedMask]] = {}
    events: list[dict] = []

    for fi in range(num_frames):
        ids, masks = per_frame.get(fi, ([], []))
        known = [(i, m) for i, m in zip(ids, masks) if i in obj_to_k]
        new = [(i, m) for i, m in zip(ids, masks) if i not in obj_to_k]
        ks_present = {obj_to_k[i] for i, _ in known}

        # try to re-associate new obj_ids with recently lost instances
        # (largest new mask first, each lost instance claimed at most once)
        new.sort(key=lambda im: -im[1].area)
        for obj_id, packed in new:
            mask = packed.unpack()
            best_k, best_iou = None, 0.0
            for k, (last_fi, last_packed) in last_seen.items():
                if k in ks_present or fi - last_fi > REASSOC_MAX_GAP:
                    continue
                iou = mask_iou(mask, last_packed.unpack())
                if iou > best_iou:
                    best_k, best_iou = k, iou
            if best_k is not None and best_iou >= REASSOC_IOU_THRESH:
                obj_to_k[obj_id] = best_k
                ks_present.add(best_k)
                events.append(
                    {
                        "frame": fi,
                        "action": "reassociate",
                        "obj_id": obj_id,
                        "instance": best_k,
                        "iou": round(best_iou, 3),
                        "gap": fi - last_seen[best_k][0],
                    }
                )
            else:
                k = len(instance_masks)
                obj_to_k[obj_id] = k
                instance_masks[k] = {}
                ks_present.add(k)

        # accumulate per-instance masks for this frame (union if an instance
        # ended up with several obj_ids visible at once)
        per_k: dict[int, np.ndarray] = {}
        for obj_id, packed in zip(ids, masks):
            k = obj_to_k[obj_id]
            m = packed.unpack()
            per_k[k] = np.logical_or(per_k[k], m) if k in per_k else m
        for k, m in per_k.items():
            pm = PackedMask(m)
            instance_masks.setdefault(k, {})[fi] = pm
            last_seen[k] = (fi, pm)

    return instance_masks, events


def render_qa_strip(
    scene: str,
    subset_dir: Path,
    frame_names: list[str],
    qa_indices: list[int],
    qa_masks: dict[str, dict[int, PackedMask]],
    out_path: Path,
) -> None:
    """One fixed color per instance dir, consistent across frames, with legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    inst_names = sorted(qa_masks.keys())
    colors = {}
    for i, name in enumerate(inst_names):
        h = (i * 0.61803398875) % 1.0
        colors[name] = colorsys.hsv_to_rgb(h, 0.85, 0.95)

    ncols = 2
    nrows = (len(qa_indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 9, nrows * 7))
    axes = np.atleast_1d(axes).ravel()

    for ax_i, fi in enumerate(qa_indices):
        ax = axes[ax_i]
        img = np.array(Image.open(subset_dir / frame_names[fi])).astype(np.float32)
        overlay = img.copy()
        for name in inst_names:
            packed = qa_masks[name].get(fi)
            if packed is None:
                continue
            m = packed.unpack()
            c = np.array(colors[name]) * 255
            overlay[m] = 0.45 * overlay[m] + 0.55 * c
            ys, xs = np.nonzero(m)
            ax.text(
                xs.mean(), ys.mean(), name,
                color="white", fontsize=7, ha="center", va="center",
                bbox=dict(facecolor=colors[name], alpha=0.8, pad=1, edgecolor="none"),
            )
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f"{scene}  frame {frame_names[fi]}", fontsize=10)
        ax.axis("off")
    for ax in axes[len(qa_indices):]:
        ax.axis("off")

    handles = [Patch(facecolor=colors[n], label=n) for n in inst_names]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(8, max(1, len(inst_names))), fontsize=8,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def is_scene_complete(out_root: Path) -> bool:
    return (out_root / ".complete").exists()


def process_scene(predictor, scene_dir: Path, qa_frame_count: int) -> dict:
    scene = scene_dir.name
    raw = scene_dir / "raw_data"
    subset_dir = raw / "subset"
    old_root = raw / "masks"
    out_root = raw / "masks_instance"

    t0 = time.time()
    frame_names = collect_frame_names(subset_dir)
    num_frames = len(frame_names)
    first = np.array(Image.open(subset_dir / frame_names[0]))
    height, width = first.shape[0], first.shape[1]
    zero_bytes = zero_png_bytes(height, width)

    qa_indices = sorted(
        {int(round(i * (num_frames - 1) / (qa_frame_count - 1))) for i in range(qa_frame_count)}
    )
    qa_masks: dict[str, dict[int, PackedMask]] = {}

    resp = predictor.handle_request(
        {"type": "start_session", "resource_path": str(subset_dir)}
    )
    session_id = resp["session_id"]

    stats = {
        "scene": scene,
        "num_frames": num_frames,
        "classes": {},
        "reassociation_events": [],
    }

    try:
        for prompt_text, class_name in CLASSES:
            resp = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt_text,
                }
            )
            per_frame = {}
            ids0, masks0 = extract_frame_objects(resp["outputs"])
            per_frame[0] = (ids0, [PackedMask(m) for m in masks0])
            per_frame.update(propagate(predictor, session_id))

            if class_name in STUFF_CLASSES:
                # single instance = union of all object masks per frame
                inst: dict[int, dict[int, PackedMask]] = {0: {}}
                seen_any = False
                for fi in range(num_frames):
                    ids, masks = per_frame.get(fi, ([], []))
                    if not ids:
                        continue
                    u = masks[0].unpack()
                    for pm in masks[1:]:
                        u |= pm.unpack()
                    inst[0][fi] = PackedMask(u)
                    seen_any = True
                instance_masks = inst if seen_any else {}
                events = []
            else:
                instance_masks, events = associate_instances(per_frame, num_frames)

            for ev in events:
                ev["class"] = class_name
            stats["reassociation_events"].extend(events)

            # write PNGs + compute union-IoU vs old per-class masks
            inter_px, union_px, frame_ious = 0, 0, []
            for fi in range(num_frames):
                stem = Path(frame_names[fi]).stem
                new_union = np.zeros((height, width), dtype=bool)
                for k, frames in instance_masks.items():
                    pm = frames.get(fi)
                    inst_dir = out_root / f"{class_name}_{k}"
                    if pm is None:
                        inst_dir.mkdir(parents=True, exist_ok=True)
                        (inst_dir / f"{stem}.png").write_bytes(zero_bytes)
                    else:
                        m = pm.unpack()
                        new_union |= m
                        save_mask_png(m, inst_dir / f"{stem}.png")

                old_path = old_root / class_name / f"{stem}.png"
                if old_path.exists():
                    old = np.array(Image.open(old_path)) > 127
                    i = int(np.logical_and(new_union, old).sum())
                    u = int(np.logical_or(new_union, old).sum())
                    inter_px += i
                    union_px += u
                    if u > 0:
                        frame_ious.append(i / u)

            for k, frames in instance_masks.items():
                for fi, pm in frames.items():
                    if fi in qa_indices:
                        qa_masks.setdefault(f"{class_name}_{k}", {})[fi] = pm

            stats["classes"][class_name] = {
                "num_instances": len(instance_masks),
                "union_iou_pixel": round(inter_px / union_px, 4) if union_px else None,
                "union_iou_frame_mean": (
                    round(float(np.mean(frame_ious)), 4) if frame_ious else None
                ),
                "frames_compared": len(frame_ious),
                "num_reassociations": len(events),
            }
            c = stats["classes"][class_name]
            print(
                f"[{scene}] {class_name}: {c['num_instances']} instances, "
                f"union-IoU(px)={c['union_iou_pixel']}, reassoc={len(events)}",
                flush=True,
            )
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    render_qa_strip(
        scene, subset_dir, frame_names, qa_indices, qa_masks,
        out_root / "_qa" / "overview.jpg",
    )

    stats["wall_clock_sec"] = round(time.time() - t0, 1)
    stats["qa_frame_indices"] = qa_indices
    qa_dir = out_root / "_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    (out_root / ".complete").write_text(json.dumps({"finished": time.time()}))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", help="Scene names (e.g. scene0000_00) or scene dirs.")
    parser.add_argument(
        "--scans_root",
        type=Path,
        default=Path("/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans"),
    )
    parser.add_argument("--qa_frames", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Re-run even if .complete exists.")
    parser.add_argument("--gpus", type=int, nargs="*", default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--log_file", type=Path, default=None,
        help="Append one JSON line per scene (stats or failure) to this file.",
    )
    args = parser.parse_args()

    scene_dirs = []
    for s in args.scenes:
        p = Path(s)
        scene_dirs.append(p if p.is_dir() else args.scans_root / s)

    gpus = range(torch.cuda.device_count()) if args.gpus is None else args.gpus
    kwargs = {"gpus_to_use": gpus}
    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    predictor = build_sam3_video_predictor(**kwargs)

    def log_line(payload: dict) -> None:
        if args.log_file:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.log_file, "a") as f:
                f.write(json.dumps(payload) + "\n")

    for scene_dir in scene_dirs:
        out_root = scene_dir / "raw_data" / "masks_instance"
        if not args.force and is_scene_complete(out_root):
            print(f"[{scene_dir.name}] already complete, skipping.", flush=True)
            continue
        try:
            stats = process_scene(predictor, scene_dir, args.qa_frames)
            total = sum(c["num_instances"] for c in stats["classes"].values())
            print(
                f"[{scene_dir.name}] DONE: {total} instances total, "
                f"{stats['wall_clock_sec']}s",
                flush=True,
            )
            log_line(stats)
        except Exception:
            print(f"[{scene_dir.name}] FAILED:\n{traceback.format_exc()}", flush=True)
            log_line({"scene": scene_dir.name, "failed": True,
                      "error": traceback.format_exc()})


if __name__ == "__main__":
    main()
