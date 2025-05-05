import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from app.config import MODEL_ID

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, processor, device