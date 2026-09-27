## My contributions

This is a fork of [facebookresearch/sam3](https://github.com/facebookresearch/sam3). All credit for
SAM 3, its training and its published results goes to Meta Superintelligence Labs. Everything below is
what I added on top; the rest of this README is theirs.

I used SAM 3 as an **automatic annotator**: to build a 2D instance-segmentation ground truth over
ScanNet scenes without manual labelling, for a 3D scene-understanding project. The work splits into a
ScanNet download and preprocessing pipeline, two mask-generation scripts (per class and per instance),
and an experiment in replacing hand-written text prompts with Florence-2 region proposals.

### What I added

**ScanNet acquisition and preprocessing pipeline**

- **[`scripts/process_scenes.sh`](scripts/process_scenes.sh)** - the end-to-end SLURM driver. For each
  scene it downloads the `.sens` archive, extracts colour frames via the ScanNet `SensReader`, renames
  them to a zero-padded 5-digit form, builds a stride-5 subset of up to 100 frames (falling back to
  however many exist for short scenes), runs mask generation, then deletes everything except
  `raw_data/`. Every step is guarded by an existence check, so the job is resumable after a timeout.
- **[`scripts/download-scannet.py`](scripts/download-scannet.py)**,
  **[`scripts/download_sens.py`](scripts/download_sens.py)** - the downloaders, with
  [`scripts/instructions_scannet.txt`](scripts/instructions_scannet.txt) recording the official terms
  and usage.
- **Second split.** [`scripts/download_split2.sh`](scripts/download_split2.sh),
  [`download_split2_manual.sh`](scripts/download_split2_manual.sh),
  [`download_split2_selfheal.sh`](scripts/download_split2_selfheal.sh) (retries failed scenes),
  [`process_split2.sh`](scripts/process_split2.sh), [`pack_split2.sh`](scripts/pack_split2.sh) and
  [`fuse_splits.sh`](scripts/fuse_splits.sh) to merge it with the first.
- **Storage management.** [`stage_to_scratch.sh`](scripts/stage_to_scratch.sh),
  [`pack_dataset.sh`](scripts/pack_dataset.sh), [`pack_and_report.sh`](scripts/pack_and_report.sh),
  [`prune_work_color.sh`](scripts/prune_work_color.sh) (drops full-resolution colour once the subset
  exists), [`cleanup_old_dataset.sh`](scripts/cleanup_old_dataset.sh),
  [`create_subset.sh`](scripts/create_subset.sh), [`rename.sh`](scripts/rename.sh).

**Mask generation**

- **[`scripts/save_text_prompt_masks.py`](scripts/save_text_prompt_masks.py)** - prompts the SAM 3
  video predictor with a fixed 19-class ScanNet taxonomy (wall, floor, cabinet, bed, chair, sofa,
  table, door, window, bookshelf, picture, counter, desk, curtain, refrigerator, shower curtain,
  toilet, sink, bathtub), propagates through the frame sequence and writes one single-channel uint8
  PNG per frame per class, values `{0, 255}`. Supports both `sam3` and `sam3.1` predictors and
  multi-GPU.
- **[`scripts/save_instance_masks.py`](scripts/save_instance_masks.py)** - the per-**instance**
  extension, and the most substantial script here. Instead of unioning all objects of a class it keeps
  SAM 3's individual masklets, tracked across frames by their persistent `out_obj_ids`, and writes
  `<scene>/raw_data/masks_instance/<class>_<k>/<frame>.png`. It adds:
  - stable instance indexing, assigned in order of first appearance and never renumbered;
  - a lost-track re-association safety net on top of SAM 3's own matching: a brand-new `obj_id` is
    compared by mask IoU against same-class instances that are currently invisible and were last seen
    within a bounded frame gap, and merged greedily best-IoU-first if it clears the threshold;
  - special handling for "stuff" classes (wall, floor), which always collapse to a single instance;
  - per-scene QA output: a colour-consistent overview strip, a `stats.json` with instance counts,
    union-IoU against the per-class masks, re-association events and wall-clock time, and a
    `.complete` marker for resumability.
  The association policy is documented in the script's own module docstring.
- **[`scripts/gen_instance_report.py`](scripts/gen_instance_report.py)** - aggregates every scene's
  `stats.json` into a single `INSTANCE_MASKS_README.md` deliverable with the layout, the policy, a
  per-scene instance-count table and the flagged or failed scenes.
- **[`scripts/run_instance_masks_bulk.sh`](scripts/run_instance_masks_bulk.sh)** - bulk driver.

**Florence-2 region proposals**

An experiment in dropping the hand-written class list: use Florence-2 to propose regions, then let
SAM 3 segment them.

- **[`sam3/florence2_utils.py`](sam3/florence2_utils.py)** - `region_proposal()` runs Florence-2-large
  with the `<REGION_PROPOSAL>` task and converts the returned absolute `[x1, y1, x2, y2]` boxes into
  the normalised `[x_min, y_min, width, height]` form SAM 3 expects. Lazily loaded and cached, with an
  `sdpa` attention path and a CPU/float32 fallback.
- **[`old/florence_unused.py`](old/florence_unused.py)** - a thinner wrapper exposing Florence-2's other task
  tokens (object detection, detailed caption, caption-to-phrase grounding, open-vocabulary detection,
  dense region caption), including the `flash_attn` import workaround needed to load the model without
  it.
- **[`scripts/debug_florence.py`](scripts/debug_florence.py)**, **[`old/test_florence_standalone.py`](old/test_florence_standalone.py)**,
  **[`examples/test_florence.ipynb`](examples/test_florence.ipynb)**,
  **[`examples/segment_with_region_proposal.ipynb`](examples/segment_with_region_proposal.ipynb)**.

**Upstream fixes needed to run this**

- **[`sam3/model/sam3_base_predictor.py`](sam3/model/sam3_base_predictor.py)** - `start_session()`
  filtered its `add_prompt` kwargs by signature but not its `init_state` kwargs, so the SAM 3.1
  multiplex predictor crashed on arguments it does not accept. Added the same
  `inspect.signature`-based filtering for `init_state`. Also removed the `torch.autocast(bfloat16)`
  wrapper around `add_prompt`.
- **[`sam3/model_builder.py`](sam3/model_builder.py)** - replaced the deprecated
  `pkg_resources.resource_filename` with `importlib.resources.files` for locating the BPE vocabulary,
  in all three builders.
- **[`sam3/model/sam3_image.py`](sam3/model/sam3_image.py)** - added type annotations to
  `_run_encoder` and `forward_video_grounding_multigpu`.
- **[`pyproject.toml`](pyproject.toml)** - added `transformers==4.41.2` for Florence-2.

**Notebooks and notes**

- [`examples/sam3_multiple_bbox_proposals_example.ipynb`](examples/sam3_multiple_bbox_proposals_example.ipynb)
  and its [`sam3_1_`](examples/sam3_1_multiple_bbox_proposals_example.ipynb) counterpart - tracking
  several objects at once from several bounding-box prompts.
- [`examples/sam3_video_bbox_prompt_example.ipynb`](examples/sam3_video_bbox_prompt_example.ipynb),
  [`examples/test_multi_prompt_video.ipynb`](examples/test_multi_prompt_video.ipynb),
  [`examples/automatic_mask_generator.ipynb`](examples/automatic_mask_generator.ipynb),
  [`old/scratch_region_proposal.ipynb`](old/scratch_region_proposal.ipynb).
- [`docs/prompt_findings.md`](docs/prompt_findings.md) - notes on prompt behaviour, in particular that generic prompts
  ("all objects", "everything") return nothing, since SAM 3 is built around atomic noun phrases, and
  that the authors recommend a structured ontology with explicit hard negatives over a flat category
  list.
- [`docs/prompt.md`](docs/prompt.md).

### Repository layout

| path | what it is |
|---|---|
| [`scripts/`](scripts/) | the ScanNet pipeline, the two mask generators, the report generator, the Florence-2 debug script |
| [`sam3/florence2_utils.py`](sam3/florence2_utils.py) | the live Florence-2 region proposer |
| [`examples/`](examples/) | the notebooks |
| [`docs/`](docs/) | prompt notes and findings |
| [`old/`](old/) | not used by the current version. See [`old/README.md`](old/README.md) |
| [`test/`](test/) | upstream's tests, untouched |

### How to run my part

```bash
# Full pipeline for a range of scenes (edit the SCENES loop and the paths at the top first):
sbatch scripts/process_scenes.sh

# Per-class masks for one already-extracted scene:
python scripts/save_text_prompt_masks.py <scene>/raw_data/color \
    --frame_step 5 --max_frames 100 --version sam3

# Per-instance masks, with the union checked against the existing per-class masks:
python scripts/save_instance_masks.py scene0097_00 scene0098_00 \
    --scans_root /path/to/scannet/scans \
    --class_masks_mode compare \
    --log_file runs.jsonl

# Aggregate the per-scene stats into the report:
python scripts/gen_instance_report.py
```

---

# SAM 3: Segment Anything with Concepts

Meta Superintelligence Labs

[Nicolas Carion](https://www.nicolascarion.com/)\*,
[Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en)\*,
[Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en)\*,
[Shoubhik Debnath](https://scholar.google.com/citations?user=fb6FOfsAAAAJ&hl=en)\*,
[Ronghang Hu](https://ronghanghu.com/)\*,
[Didac Suris](https://www.didacsuris.com/)\*,
[Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en)\*,
[Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en)\*,
[Haitham Khedr](https://hkhedr.com/)\*, Andrew Huang,
[Jie Lei](https://jayleicn.github.io/),
[Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en),
[Baishan Guo](https://scholar.google.com/citations?user=BC5wDu8AAAAJ&hl=en),
Arpit Kalla, [Markus Marks](https://damaggu.github.io/),
[Joseph Greer](https://scholar.google.com/citations?user=guL96CkAAAAJ&hl=en),
Meng Wang, [Peize Sun](https://peizesun.github.io/),
[Roman Rädle](https://scholar.google.com/citations?user=Tpt57v0AAAAJ&hl=en),
[Triantafyllos Afouras](https://www.robots.ox.ac.uk/~afourast/),
[Effrosyni Mavroudi](https://scholar.google.com/citations?user=vYRzGGEAAAAJ&hl=en),
[Katherine Xu](https://k8xu.github.io/)°,
[Tsung-Han Wu](https://patrickthwu.com/)°,
[Yu Zhou](https://yu-bryan-zhou.github.io/)°,
[Liliane Momeni](https://scholar.google.com/citations?user=Lb-KgVYAAAAJ&hl=en)°,
[Rishi Hazra](https://rishihazra.github.io/)°,
[Shuangrui Ding](https://mark12ding.github.io/)°,
[Sagar Vaze](https://sgvaze.github.io/)°,
[Francois Porcher](https://scholar.google.com/citations?user=LgHZ8hUAAAAJ&hl=en)°,
[Feng Li](https://fengli-ust.github.io/)°,
[Siyuan Li](https://siyuanliii.github.io/)°,
[Aishwarya Kamath](https://ashkamath.github.io/)°,
[Ho Kei Cheng](https://hkchengrex.com/)°,
[Piotr Dollar](https://pdollar.github.io/)†,
[Nikhila Ravi](https://nikhilaravi.com/)†,
[Kate Saenko](https://ai.bu.edu/ksaenko.html)†,
[Pengchuan Zhang](https://pzzhang.github.io/pzzhang/)†,
[Christoph Feichtenhofer](https://feichtenhofer.github.io/)†

\* core contributor, ° intern, † project lead, order is random within groups

[[`Paper`](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)]
[[`Project`](https://ai.meta.com/sam3)]
[[`Demo`](https://segment-anything.com/)]
[[`Blog`](https://ai.meta.com/blog/segment-anything-model-3/)]
[[`BibTeX`](#citing-sam-3)]

![SAM 3 architecture](assets/model_diagram.png?raw=true) SAM 3 is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor [SAM 2](https://github.com/facebookresearch/sam2), SAM 3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase or exemplars. Unlike prior work, SAM 3 can handle a vastly larger set of open-vocabulary prompts. It achieves 75-80% of human performance on our new [SA-CO benchmark](https://github.com/facebookresearch/sam3?tab=readme-ov-file#sa-co-dataset) which contains 270K unique concepts, over 50 times more than existing benchmarks.

This breakthrough is driven by an innovative data engine that has automatically annotated over 4 million unique concepts, creating the largest high-quality open-vocabulary segmentation dataset to date. In addition, SAM 3 introduces a new model architecture featuring a presence token that improves discrimination between closely related text prompts (e.g., “a player in white” vs. “a player in red”), as well as a decoupled detector–tracker design that minimizes task interference and scales efficiently with data.

<p align="center">
  <img src="assets/dog.gif" width=380 />
  <img src="assets/player.gif" width=380 />
</p>

## Latest updates

**03/27/2026 -- SAM 3.1 Object Multiplex is released. It introduces a shared-memory approach for joint multi-object tracking that is significantly faster without sacrificing accuracy.**

- A new suite of improved model checkpoints (denoted as **SAM 3.1**) are released on [Hugging Face](https://huggingface.co/facebook/sam3.1). See [`RELEASE_SAM3p1.md`](RELEASE_SAM3p1.md) for full details.
  * To use the new SAM 3.1 checkpoints, you need the latest model code from this repo. If you have installed an earlier version of this repo, pull the latest code from this repo (with `git pull`), and then reinstall the repo following [Installation](#installation) below.

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher

1. **Create a new Conda environment:**

```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
```

2. **Install PyTorch with CUDA support:**

```bash
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

3. **Clone the repository and install the package:**

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

4. **Install additional dependencies for example notebooks or development:**

```bash
# For running example notebooks
pip install -e ".[notebooks]"

# For development
pip install -e ".[train,dev]"
```

5. **Optional dependencies for faster inference**
```bash
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git
```

## Getting Started

⚠️ Before using SAM 3, please request access to the checkpoints on the SAM 3
Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)

### Basic Usage

```python
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#################################### For Video ####################################

from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]
```

## Examples

The `examples` directory contains notebooks demonstrating how to use SAM3 with
various types of prompts:

- [`sam3_image_predictor_example.ipynb`](examples/sam3_image_predictor_example.ipynb)
  : Demonstrates how to prompt SAM 3 with text and visual box prompts on images.
- [`sam3_video_predictor_example.ipynb`](examples/sam3_video_predictor_example.ipynb)
  : Demonstrates how to prompt SAM 3 with text prompts on videos, and doing
  further interactive refinements with points.
- [`sam3_image_batched_inference.ipynb`](examples/sam3_image_batched_inference.ipynb)
  : Demonstrates how to run batched inference with SAM 3 on images.
- [`sam3_agent.ipynb`](examples/sam3_agent.ipynb): Demonsterates the use of SAM
  3 Agent to segment complex text prompt on images.
- [`saco_gold_silver_vis_example.ipynb`](examples/saco_gold_silver_vis_example.ipynb)
  : Shows a few examples from SA-Co image evaluation set.
- [`saco_veval_vis_example.ipynb`](examples/saco_veval_vis_example.ipynb) :
  Shows a few examples from SA-Co video evaluation set.

There are additional notebooks in the examples directory that demonstrate how to
use SAM 3 for interactive instance segmentation in images and videos (SAM 1/2
tasks), or as a tool for an MLLM, and how to run evaluations on the SA-Co
dataset.

To run the Jupyter notebook examples:

```bash
# Make sure you have the notebooks dependencies installed
pip install -e ".[notebooks]"

# Start Jupyter notebook
jupyter notebook examples/sam3_image_predictor_example.ipynb
```

## Model

SAM 3 consists of a detector and a tracker that share a vision encoder. It has 848M parameters. The
detector is a DETR-based model conditioned on text, geometry, and image
exemplars. The tracker inherits the SAM 2 transformer encoder-decoder
architecture, supporting video segmentation and interactive refinement.

## Image Results

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="3" style="border-right: 2px solid #ddd; padding: 12px 20px">Model</th>
      <th colspan="3" style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">Instance Segmentation</th>
      <th colspan="5" style="text-align: center; padding: 12px 20px">Box Detection</th>
    </tr>
    <tr>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVIS</th>
      <th style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">SA-Co/Gold</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVIS</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">COCO</th>
      <th style="text-align: center; padding: 12px 20px">SA-Co/Gold</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP</th>
      <th style="text-align: center; border-right: 2px solid #ddd; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP</th>
      <th style="text-align: center; padding: 12px 20px">AP</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">AP<sub>o</sub>
</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">Human</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">72.8</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">74.0</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">OWLv2*</td>
      <td style="text-align: center; padding: 10px 20px; color: #999">29.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px; color: #999">43.4</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">24.6</td>
      <td style="text-align: center; padding: 10px 20px; color: #999">30.2</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px; color: #999">45.5</td>
      <td style="text-align: center; padding: 10px 20px">46.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">23.9</td>
      <td style="text-align: center; padding: 10px 20px">24.5</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">DINO-X</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">38.5</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">21.3</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">52.4</td>
      <td style="text-align: center; padding: 10px 20px">56.0</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">22.5</td>
    </tr>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">Gemini 2.5</td>
      <td style="text-align: center; padding: 10px 20px">13.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">13.0</td>
      <td style="text-align: center; padding: 10px 20px">16.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">14.4</td>
    </tr>
    <tr style="border-top: 2px solid #b19c9cff">
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">SAM 3</td>
      <td style="text-align: center; padding: 10px 20px">37.2</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">48.5</td>
      <td style="text-align: center; border-right: 2px solid #ddd; padding: 10px 20px">54.1</td>
      <td style="text-align: center; padding: 10px 20px">40.6</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">53.6</td>
      <td style="text-align: center; padding: 10px 20px">56.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">55.7</td>
      <td style="text-align: center; padding: 10px 20px">55.7</td>
    </tr>
  </tbody>
</table>

<p style="text-align: center; margin-top: 10px; font-size: 0.9em; color: #ddd;">* Partially trained on LVIS, AP<sub>o</sub> refers to COCO-O accuracy</p>

</div>

## Video Results

<div align="center">
<table style="min-width: 80%; border: 2px solid #ddd; border-collapse: collapse">
  <thead>
    <tr>
      <th rowspan="2" style="border-right: 2px solid #ddd; padding: 12px 20px">Model</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">SA-V test</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">YT-Temporal-1B test</th>
      <th colspan="2" style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">SmartGlasses test</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">LVVIS test</th>
      <th style="text-align: center; padding: 12px 20px">BURST test</th>
    </tr>
    <tr>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; padding: 12px 20px">cgF1</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">pHOTA</th>
      <th style="text-align: center; border-right: 1px solid #eee; padding: 12px 20px">mAP</th>
      <th style="text-align: center; padding: 12px 20px">HOTA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">Human</td>
      <td style="text-align: center; padding: 10px 20px">53.1</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">70.5</td>
      <td style="text-align: center; padding: 10px 20px">71.2</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">78.4</td>
      <td style="text-align: center; padding: 10px 20px">58.5</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">72.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">-</td>
      <td style="text-align: center; padding: 10px 20px">-</td>
    </tr>
    <tr style="border-top: 2px solid #b19c9cff">
      <td style="border-right: 2px solid #ddd; padding: 10px 20px">SAM 3</td>
      <td style="text-align: center; padding: 10px 20px">30.3</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">58.0</td>
      <td style="text-align: center; padding: 10px 20px">50.8</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">69.9</td>
      <td style="text-align: center; padding: 10px 20px">36.4</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">63.6</td>
      <td style="text-align: center; border-right: 1px solid #eee; padding: 10px 20px">36.3</td>
      <td style="text-align: center; padding: 10px 20px">44.5</td>
    </tr>
  </tbody>
</table>
</div>

## SA-Co Dataset

We release 2 image benchmarks, [SA-Co/Gold](scripts/eval/gold/README.md) and
[SA-Co/Silver](scripts/eval/silver/README.md), and a video benchmark
[SA-Co/VEval](scripts/eval/veval/README.md). The datasets contain images (or videos) with annotated noun phrases. Each image/video and noun phrase pair is annotated with instance masks and unique IDs of each object matching the phrase. Phrases that have no matching objects (negative prompts) have no masks, shown in red font in the figure. See the linked READMEs for more details on how to download and run evaluations on the datasets.

* HuggingFace host: [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold), [SA-Co/Silver](https://huggingface.co/datasets/facebook/SACo-Silver) and [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval)
* Roboflow host: [SA-Co/Gold](https://universe.roboflow.com/sa-co-gold), [SA-Co/Silver](https://universe.roboflow.com/sa-co-silver) and [SA-Co/VEval](https://universe.roboflow.com/sa-co-veval)

![SA-Co dataset](assets/sa_co_dataset.jpg?raw=true)

## Development

To set up the development environment:

```bash
pip install -e ".[dev,train]"
```

To format the code:

```bash
ufmt format .
```

## Contributing

See [contributing](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file
for details.

## Acknowledgements

We would like to thank the following people for their contributions to the SAM 3 project: Alex He, Alexander Kirillov,
Alyssa Newcomb, Ana Paula Kirschner Mofarrej, Andrea Madotto, Andrew Westbury, Ashley Gabriel, Azita Shokpour,
Ben Samples, Bernie Huang, Carleigh Wood, Ching-Feng Yeh, Christian Puhrsch, Claudette Ward, Daniel Bolya,
Daniel Li, Facundo Figueroa, Fazila Vhora, George Orlin, Hanzi Mao, Helen Klein, Hu Xu, Ida Cheng, Jake Kinney,
Jiale Zhi, Jo Sampaio, Joel Schlosser, Justin Johnson, Kai Brown, Karen Bergan, Karla Martucci, Kenny Lehmann,
Maddie Mintz, Mallika Malhotra, Matt Ward, Michelle Chan, Michelle Restrepo, Miranda Hartley, Muhammad Maaz,
Nisha Deo, Peter Park, Phillip Thomas, Raghu Nayani, Rene Martinez Doehner, Robbie Adkins, Ross Girshik, Sasha
Mitts, Shashank Jain, Spencer Whitehead, Ty Toledano, Valentin Gabeur, Vincent Cho, Vivian Lee, William Ngan,
Xuehai He, Yael Yungster, Ziqi Pang, Ziyi Dou, Zoe Quake.

## Citing SAM 3

If you use SAM 3 or the SA-Co dataset in your research, please use the following BibTeX entry.

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```
