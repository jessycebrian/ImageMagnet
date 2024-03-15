from clip import processor,model
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from pinecone_connection import right_index

def get_image_from_url(image_url):
    """
    Fetches an image from the provided URL and returns a PIL Image object.

    Parameters:
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
    Fetches image urls from vectorDB result.

    Parameters:
        image_url (str): URL of the image.

    Returns:
        Image: image url list object.
    """
    images = []
    if 'matches' in result:
        for match in result['matches']:
            if 'metadata' in match and 'image' in match['metadata']:
                image_url = match['metadata']['image']
                images.append(image_url)
    return images

def get_images_from_query(text:str,top_k:int = 5):
    inputs = processor.tokenizer(text, max_length=77, truncation=True, padding="max_length", return_tensors="pt")
    embedding = model.get_text_features(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])
    results = right_index.query(vector = embedding.detach().numpy().tolist(), top_k=top_k, include_metadata=True)
    return display_images_from_vectordb_result(results)