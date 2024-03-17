import logging


import pandas as pd
import torch

from datasets.clipdataset import CLIPDataset
from pinecone_connection import right_index
from transformers import CLIPModel, CLIPProcessor
from utils.embeddings import get_all_embeddings
from utils.similarities import get_similarities


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CLIP2Vdb:
    HISTOGRAM_SAMPLES = 472
    def __init__(
        self, model: CLIPModel, processor: CLIPProcessor, device: str = "cpu", vdb_index = right_index
    ) -> None:
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.vdb_index = vdb_index
        self.text_embeddings = []
        self.image_embeddings = []

    def get_embeddings(self,dataset: CLIPDataset):
        logging.info("Getting Embeddings ...")
        _, self.text_embeddings, self.image_embeddings = get_all_embeddings(dataset, self.model, self.processor, self.device)
        logging.info("Embeddings done!")
        logging.info(f"Embeddings shape - Images: {self.image_embeddings.shape}, Text: {self.text_embeddings.shape}!")
    
    def _normalise_embeddings(self):
        # self.image_embeddings = torch.nn.functional.normalize(self.image_embeddings, p=2, dim=1)
        # self.text_embeddings = torch.nn.functional.normalize(self.text_embeddings, p=2, dim=1)
        self.image_embeddings = self.image_embeddings
        self.text_embeddings = self.text_embeddings

    def _combine_text_image_embeddings(self):
        """Compute average of text and image embeddings."""
        # self.final_embeddings = torch.mean(torch.stack([self.image_embeddings, self.text_embeddings]), dim=0)
        self.final_embeddings = self.image_embeddings

    def _upload_data_to_index(self, df):
        logging.info("Uploading Embeddings to vector Database...")
    # Convert index to string
        df["vector_id"] = df.index
        df["vector_id"] = df["vector_id"].apply(str)
        # Get all the metadata
        final_metadata = []
        for index in range(len(df)):
            final_metadata.append({
                'ID':  index,
                'query': df.iloc[index].queries,
                'image': df.iloc[index].url
            })
        image_IDs = df.vector_id.tolist()
        # Create the single list of dictionary format to insert
        data_to_upsert = list(zip(image_IDs, self.final_embeddings, final_metadata))
        # Upload the final data
        self.vdb_index.upsert(data_to_upsert)
        logging.info("Embeddings uploaded successfully.")

    def upsert_embeddings(self, df:pd.DataFrame):
        logging.info("Normalising Embeddings ...")
        self._normalise_embeddings()
        logging.info("Combining text and image Embeddings ...")
        self._combine_text_image_embeddings()
        self._upload_data_to_index(df)

    def get_index_stats(self):
        return self.vdb_index.describe_index_stats()

    def measure_embeddings_similarities(self,embeddings:torch.Tensor):
        logging.info("Getting embeddings:")
        logging.info("Embeddings done!")
        logging.info(f"Embeddings shape - Images: {embeddings.shape}, Text: {self.text_embeddings.shape}!")
        logging.info("Getting similarities")
        similarities = get_similarities(
            text_embeddings=self.text_embeddings, image_embeddings=embeddings, sample_size=self.HISTOGRAM_SAMPLES
        )
        logging.info(f"Similarities shape: {similarities.shape}")
        return similarities

