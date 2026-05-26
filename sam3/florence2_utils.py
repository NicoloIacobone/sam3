from typing import Union
from PIL import Image

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
        import os

        model_id = 'microsoft/Florence-2-large'

        os.environ["FLASH_ATTENTION_SKIP_TORCH_CHECK"] = "True"

        try:
            _model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa"
            ).eval()
        except Exception as e:
            print(f"Failed to load with sdpa attention: {e}")
            print("Trying with default attention...")
            _model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu"
            ).eval()

        _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return _model, _processor


def region_proposal(image: Union[str, Image.Image]) -> dict:
    """
    Extract region proposals (bounding boxes) from an image using Florence2.

    Args:
        image: PIL Image or path to image file

    Returns:
        Dict with 'bboxes' (in SAM format [xmin, ymin, width, height]) and 'labels' keys
    """
    if isinstance(image, str):
        image = Image.open(image)

    model, processor = _load_model()
    prompt = '<REGION_PROPOSAL>'

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids,
                                            skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height))

    result = parsed[prompt]

    # Convert bboxes from [x1, y1, x2, y2] (absolute) to [xmin, ymin, width, height] (normalized)
    if 'bboxes' in result:
        converted_bboxes = []
        for x1, y1, x2, y2 in result['bboxes']:
            x_min = x1 / image.width
            y_min = y1 / image.height
            width = (x2 - x1) / image.width
            height = (y2 - y1) / image.height
            converted_bboxes.append([x_min, y_min, width, height])
        result['bboxes'] = converted_bboxes

    return result
