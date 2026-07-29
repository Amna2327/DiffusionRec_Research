from unsloth import FastVisionModel
from transformers import AutoModel

model, tokenizer = FastVisionModel.from_pretrained(
    "nomypython/urdu-ocr-deepseek",
    load_in_4bit=True,
    auto_model=AutoModel,
    trust_remote_code=True,
)

FastVisionModel.for_inference(model)

result = model.infer(
    tokenizer,
    prompt="<image>\nExtract Urdu text from this image:",
    image_file=r"C:\Users\sa2\sarmad\Urdu_Handwriting_Word_Generation\Urdu_Word_Dataset\test\processed_images\3.jpg",
    image_size=64,
    base_size=64,
    crop_mode=False,
)

print(result)
