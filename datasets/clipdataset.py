import base64
from io import BytesIO

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode


class CLIPDataset(Dataset):
    """
    A PyTorch Dataset class for CLIP model inputs, containing image and text pairs.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the CLIPDataset.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing 'queries' and 'image_bytes' columns.
        """
        self.queries = df["queries"]
        self.images = df["image_bytes"]
        self.transform = transforms.Compose([
            Resize([224], interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
        ])

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the image and its corresponding query.
        """
        image = self.transform(Image.open(BytesIO(base64.b64decode(self.images[idx]))))
        query = self.queries[idx]
        return {"image": image, "query": query}

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.queries)
