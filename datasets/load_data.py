import logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd 
from utils.utils import url_to_base64

def load_data(path:str = 'sample_property_images.csv', debug = None):
    """
    Loads property image data from a CSV file.

    Args:
        path (str): Path to the CSV file. Default is 'sample_property_images.csv'.
        debug (int, optional): Number of rows to load for debugging. Default is None.

    Returns:
        pandas.DataFrame: DataFrame containing the loaded data.
    """
    property_images = pd.read_csv(path)
    if debug:
        property_images = property_images[:debug] # for debugging
    logging.info(f"Dataset with len {len(property_images.queries)} read")

    logging.info(f"Transforming URL to image bytes ... ")
    # transform url to image bytes 
    property_images['image_bytes'] = property_images['url'].apply(lambda x: url_to_base64(x))
     # remove images that images return an error.
    property_images = property_images.loc[property_images['image_bytes'] != 'Error'].reset_index(drop=True)
    return property_images
