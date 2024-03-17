from utils import processor, model
from PIL import Image
import requests
from io import BytesIO
import base64
from pinecone_connection import right_index
import matplotlib.pyplot as plt

def url_to_base64(url: str):
    """
    Convert an image from a URL to base64 format.

    Args:
        url (str): The URL of the image.

    Returns:
        str: Base64-encoded image data.
    """
    try:
        url = url.format(transformations='w_224,h_224')
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format='JPEG')
        results = base64.b64encode(buffered.getvalue())
    except Exception as e:
        results = 'Error'
    return results

def get_image_from_url(image_url):
    """
    Fetches an image from the provided URL and returns a PIL Image object.

    Args:
        image_url (str): URL of the image.

    Returns:
        Image: PIL Image object.
    """
    try:
        image_url = image_url.format(transformations='w_224,h_224')
        response = requests.get(image_url)
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
    except Exception as e:
        print(e)
        image = None

    return image

def display_images_from_vectordb_result(result):
    """
    Fetches image URLs from a VectorDB result.

    Args:
        result (dict): Result of a VectorDB query.

    Returns:
        list: List of image URLs.
    """
    images = []
    if 'matches' in result:
        for match in result['matches']:
            if 'metadata' in match and 'image' in match['metadata']:
                image_url = match['metadata']['image']
                images.append(image_url)
    return images

def get_images_from_query(text:str, top_k:int = 5):
    """
    Retrieves image URLs related to a text query.

    Args:
        text (str): Text query.
        top_k (int): Number of top matches to retrieve.

    Returns:
        list: List of image URLs.
    """
    inputs = processor.tokenizer(text, max_length=77, truncation=True, padding="max_length", return_tensors="pt")
    embedding = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    results = right_index.query(vector=embedding.detach().numpy().tolist(), top_k=top_k, include_metadata=True)
    return display_images_from_vectordb_result(results)

def get_ids_from_vectordb_result(result):
    """
    Extracts IDs from a VectorDB result.

    Args:
        result (dict): Result of a VectorDB query.

    Returns:
        list: List of IDs.
    """
    ids = []
    if 'matches' in result:
        for match in result['matches']:
            if 'metadata' in match and 'ID' in match['metadata']:
                id = match['metadata']['ID']
                ids.append(int(id))
    return ids

def get_ids_from_query(text:str, top_k:int = 5):
    """
    Retrieves IDs related to a text query.

    Args:
        text (str): Text query.
        top_k (int): Number of top matches to retrieve.

    Returns:
        list: List of IDs.
    """
    inputs = processor.tokenizer(text, max_length=77, truncation=True, padding="max_length", return_tensors="pt")
    embedding = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    results = right_index.query(vector=embedding.detach().numpy().tolist(), top_k=top_k, include_metadata=True)
    return get_ids_from_vectordb_result(results)
