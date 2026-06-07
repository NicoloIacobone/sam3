"""Save SAM3 text-prompt masks next to an image folder.

This script mirrors the notebook flow:
1. Build a video predictor.
2. Start a session on an image folder.
3. Prompt the model with a list of class names.
4. Propagate through the video.
5. Save one binary mask per frame and class into a sibling masks folder.

The saved masks are single-channel uint8 PNG images with values 0 and 255.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_predictor


CLASSES = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")


def collect_frame_names(image_dir: Path) -> list[str]:
    frame_names = [
        entry.name
        for entry in sorted(image_dir.iterdir())
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not frame_names:
        raise RuntimeError(f"No image frames found in {image_dir}")
    return frame_names


def sample_frame_names(frame_names: list[str], max_frames: int, frame_step: int) -> list[str]:
    sampled = frame_names[::frame_step][:max_frames]
    return sampled if sampled else frame_names[:1]


def create_sampled_frame_dir(image_dir: Path, frame_names: list[str]) -> Path:
    sampled_dir = Path(tempfile.mkdtemp(prefix="sam3_sampled_frames_"))
    for frame_name in frame_names:
        src = image_dir / frame_name
        dst = sampled_dir / frame_name
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return sampled_dir


def load_frame_size(frame_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(frame_path))
    if image is None:
        raise RuntimeError(f"Failed to read frame {frame_path}")
    return image.shape[0], image.shape[1]


def make_union_mask(outputs: dict, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    mask = np.zeros((height, width), dtype=bool)

    obj_masks = outputs.get("out_binary_masks")
    if obj_masks is None:
        return mask.astype(np.uint8) * 255

    if isinstance(obj_masks, torch.Tensor):
        obj_masks = obj_masks.detach().cpu().numpy()

    for obj_mask in obj_masks:
        if obj_mask.ndim == 3:
            obj_mask = obj_mask[0]
        mask |= obj_mask.astype(bool)

    return mask.astype(np.uint8) * 255


def save_mask_png(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path)


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session_id}
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM3 text prompts on an image folder and save masks."
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Folder containing the input frames.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to save masks. Defaults to a sibling masks/ folder.",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=5,
        help="Sample every Nth frame before sending them to the predictor.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum number of frames to keep after sampling.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="sam3",
        choices=["sam3", "sam3.1"],
        help="Predictor version to build.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional local checkpoint path.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=None,
        help="GPU indices to use. Defaults to all available GPUs.",
    )
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = image_dir.parent / "masks"
    output_dir = output_dir.resolve()

    frame_names = collect_frame_names(image_dir)
    sampled_frame_names = sample_frame_names(frame_names, args.max_frames, args.frame_step)
    sampled_dir = create_sampled_frame_dir(image_dir, sampled_frame_names)
    frame_sizes = [load_frame_size(sampled_dir / frame_name) for frame_name in sampled_frame_names]

    try:
        gpus_to_use = range(torch.cuda.device_count()) if args.gpus is None else args.gpus
        predictor_kwargs = {"gpus_to_use": gpus_to_use}
        if args.checkpoint is not None:
            predictor_kwargs["checkpoint_path"] = args.checkpoint
        predictor = build_sam3_video_predictor(**predictor_kwargs)

        response = predictor.handle_request(
            {"type": "start_session", "resource_path": str(sampled_dir)}
        )
        session_id = response["session_id"]

        try:
            predictor.handle_request({"type": "reset_session", "session_id": session_id})

            for class_name in tqdm(CLASSES, desc="class prompts"):
                response = predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": class_name,
                    }
                )

                outputs_by_frame = {0: response["outputs"]}
                outputs_by_frame.update(propagate_in_video(predictor, session_id))

                class_dir = output_dir / sanitize_name(class_name)
                for frame_idx, frame_name in enumerate(sampled_frame_names):
                    outputs = outputs_by_frame.get(frame_idx, {})
                    mask = make_union_mask(outputs, frame_sizes[frame_idx])
                    save_mask_png(mask, class_dir / f"{Path(frame_name).stem}.png")

                print(f"Saved masks for '{class_name}' into {class_dir}")
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
    finally:
        shutil.rmtree(sampled_dir, ignore_errors=True)

    print(f"Mask root: {output_dir}")
    print("Stored masks as uint8 PNGs with 0 for background and 255 for foreground.")


if __name__ == "__main__":
    main()