# old/

Not used by the current version, kept for the record.

| path | what it was | why it is here |
|---|---|---|
| `florence_unused.py` | wrapper exposing Florence-2's task tokens (OD, captioning, grounding, open-vocabulary detection) | nothing imports it; the live region proposer is `sam3/florence2_utils.py` |
| `test_florence_standalone.py` | self-contained Florence-2 smoke test | superseded by `examples/segment_with_region_proposal.ipynb` |
| `scratch_region_proposal.ipynb` | scratch notebook while wiring up region proposals | superseded by `examples/segment_with_region_proposal.ipynb` |
| `sam3_video_bbox_prompt_example_copy.ipynb` | stray duplicate | duplicate of `examples/sam3_video_bbox_prompt_example.ipynb` |
| `sam3_video_predictor_example_upstream.ipynb` | upstream's notebook as it was before I edited it | kept as the reference point for the diff |
