from itertools import chain
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from datasets.clipdataset import CLIPDataset

def get_image_embedding(image, model: CLIPModel, processor: CLIPProcessor, device: str):
    """
    Get the image embedding using the CLIP model.

    Args:
        image (ndarray): The image data.
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.
        device (str): The device to perform computations on.

    Returns:
        Tensor: The image embedding.
    """
    image = processor(text=None, images=image, return_tensors="pt", do_rescale=False)["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    return embedding.cpu().detach()

def get_text_embedding(text, model: CLIPModel, processor: CLIPProcessor, device: str):
    """
    Get the text embedding using the CLIP model.

    Args:
        text (str): The input text.
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.
        device (str): The device to perform computations on.

    Returns:
        Tensor: The text embedding.
    """
    # Tokenize the input text, trim to max input length for clip
    input = processor.tokenizer(text, max_length=77, truncation=True, padding="max_length", return_tensors="pt").to(device)
    text_features = model.to(device).get_text_features(**input)
    return text_features.cpu().detach()


def get_all_embeddings(dataset: CLIPDataset, model: CLIPModel, processor: CLIPProcessor, device: str):
    """
    Get all embeddings for a given dataset.

    Args:
        dataset (CLIPDataset): The dataset containing images and queries.
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.
        device (str): The device to perform computations on.

    Returns:
        Tuple: A tuple containing queries, text embeddings, and image embeddings.
    """
    image_embeddings = []
    text_embeddings = []
    queries = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in dataloader:
        image_embeddings.append(get_image_embedding(data["image"], model, processor, device))
        text_embeddings.append(get_text_embedding(data["query"], model, processor, device))
        queries.append(data["query"])

    queries = list(chain.from_iterable(queries))
    text_embeddings = torch.cat(text_embeddings)
    image_embeddings = torch.cat(image_embeddings)

    return queries, text_embeddings, image_embeddings
