import os
os.environ["FLASH_ATTENTION_SKIP_WORKSPACE_CHECK"] = "1"

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load with SDPA attention
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    attn_implementation="sdpa"
).to(device)

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)

def get_bbox_proposals(image):
    prompt = "<REGION_PROPOSAL>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task="<REGION_PROPOSAL>",
        image_size=(image.width, image.height)
    )
    
    return parsed_answer["<REGION_PROPOSAL>"]["bboxes"]

# Test
image = Image.open("test.jpg")
bboxes = get_bbox_proposals(image)
print(bboxes)