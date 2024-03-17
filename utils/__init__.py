from transformers import CLIPProcessor, CLIPModel
import torch

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

_ = model.to(device)