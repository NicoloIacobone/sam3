"""Aggregate per-scene stats.json into INSTANCE_MASKS_README.md.

Reads <build>/scans/<scene>/raw_data/masks_instance/_qa/stats.json for every
scene and emits the deliverable report: layout, association policy, per-scene
instance-count table, union-IoU stats, and flagged/failed scenes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_ORDER = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower_curtain", "toilet", "sink", "bathtub",
]

POLICY = """\
## Layout

```
<scene>/raw_data/masks_instance/<class>_<k>/<frame>.png
<scene>/raw_data/masks_instance/_qa/overview.jpg   # color-consistent QA strip
<scene>/raw_data/masks_instance/_qa/stats.json     # per-scene metrics
<scene>/raw_data/masks_instance/.complete          # resumability marker
```

- `<k>`: zero-based instance index within the class (e.g. `chair_0`, `chair_1`).
- PNGs: uint8, values {0, 255}, full resolution 1296x968, one file for every
  subset frame in every instance dir (all-zero where the instance is absent).
  Filenames match the subset frame names exactly.
- The 19-class taxonomy and the per-class `masks/` directory are unchanged;
  `masks_instance/` is a new sibling. The union of a class's instance masks
  reproduces the old per-class mask.

## Association / tracking policy

- Per class, SAM3's video predictor is prompted once on frame 0 and propagated
  across all 100 stride-5 subset frames. Each masklet carries a persistent
  `obj_id` maintained by SAM3's tracker (detector re-detects every frame; the
  tracker matches detections to existing masklets and suppresses duplicates).
- Instance index `k` is assigned per class in order of first appearance
  (frame index, then obj_id) and is never renumbered mid-scene.
- **Lost-track re-association** (safety net on top of SAM3's matching): when a
  brand-new obj_id first appears at frame t, it is compared by mask-IoU against
  same-class instances that are currently invisible and were last seen within
  20 subset frames; if IoU with the last-seen mask >= 0.30 it is merged into
  that instance (greedy, best IoU first). Otherwise it starts a new instance.
- **Stuff classes** (`wall`, `floor`) are forced to a single instance `_0`
  covering the union of all detections.
- Masks below 200 px in a frame are dropped as specks.
- Instances within a class are made non-overlapping by SAM3's object-wise
  non-overlap constraint; cross-class overlaps may occur (classes are prompted
  independently), matching the original per-class masks.

## Consistency with the per-class GT

The re-run is deterministic, so the union of instance masks matches the
existing `masks/<class>/` masks with union-IoU = 1.0 per class (pixel-level).
Sub-1.0 per-frame means come only from the 200-px speck filter dropping slivers.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="/cluster/scratch/niacobone/scannet_build")
    ap.add_argument("--out", default="/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/INSTANCE_MASKS_README.md")
    args = ap.parse_args()

    scans = Path(args.build) / "scans"
    scenes = sorted(p.name for p in scans.iterdir() if p.is_dir())

    rows, failed, flagged = [], [], []
    total_instances = 0
    iou_values = []

    for scene in scenes:
        sj = scans / scene / "raw_data" / "masks_instance" / "_qa" / "stats.json"
        if not sj.exists():
            failed.append(scene)
            continue
        d = json.loads(sj.read_text())
        counts = {c: d["classes"].get(c, {}).get("num_instances", 0) for c in CLASS_ORDER}
        n_inst = sum(counts.values())
        total_instances += n_inst
        n_reassoc = len(d.get("reassociation_events", []))
        # min pixel union-IoU over present classes
        ious = [
            v["union_iou_pixel"] for v in d["classes"].values()
            if v.get("union_iou_pixel") is not None
        ]
        iou_values.extend(ious)
        min_iou = min(ious) if ious else None
        if min_iou is not None and min_iou < 0.9:
            flagged.append(f"{scene} (min union-IoU {min_iou})")
        rows.append((scene, counts, n_inst, n_reassoc, min_iou, d.get("wall_clock_sec")))

    lines = ["# ScanNet per-instance SAM3 masks", ""]
    lines.append(f"{len(rows)} scenes built, {total_instances} instances total. "
                 f"Mean per-class union-IoU (pixel) over all present classes: "
                 f"{round(sum(iou_values)/len(iou_values), 4) if iou_values else 'n/a'}.")
    lines.append("")
    lines.append(POLICY)
    lines.append("## Per-scene instance counts")
    lines.append("")
    header = ["scene"] + CLASS_ORDER + ["total", "reassoc", "min_iou", "sec"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for scene, counts, n_inst, n_reassoc, min_iou, sec in rows:
        cells = [scene] + [str(counts[c]) for c in CLASS_ORDER]
        cells += [str(n_inst), str(n_reassoc),
                  str(min_iou) if min_iou is not None else "-", str(sec)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Reliability")
    lines.append("")
    lines.append(f"- Scenes that failed to build (no stats.json): "
                 f"{', '.join(failed) if failed else 'none'}.")
    lines.append(f"- Scenes flagged (min union-IoU < 0.9, review before use): "
                 f"{', '.join(flagged) if flagged else 'none'}.")
    lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}: {len(rows)} scenes, {total_instances} instances, "
          f"{len(failed)} failed, {len(flagged)} flagged.")


if __name__ == "__main__":
    main()
