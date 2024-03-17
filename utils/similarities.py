import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

def get_similarities(text_embeddings: torch.Tensor, image_embeddings: torch.Tensor, sample_size: int = None) -> pd.DataFrame:
    """
    Computes similarities between text and image embeddings.

    Args:
        text_embeddings (torch.Tensor): Tensor of text embeddings.
        image_embeddings (torch.Tensor): Tensor of image embeddings.
        sample_size (int, optional): Number of samples to take from the computed similarities. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing similarities between text and image embeddings.
    """
    similarities = (text_embeddings @ image_embeddings.T).detach().numpy()
    
    df_similarities = pd.DataFrame(
        {"similarity": similarities.flatten(), "label": np.identity(similarities.shape[0]).flatten()}
    )
    if sample_size:
        sampled_df = df_similarities.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size))
        )
        return sampled_df
    return df_similarities

def plot_similarities(df_similarities: pd.DataFrame, sample_by_group: int = 30, filename: str = 'plot.png') -> None:
    """
    Plots the distribution of similarities between positive and negative examples.

    Args:
        df_similarities (pd.DataFrame): DataFrame containing similarities.
        sample_by_group (int, optional): Number of samples per group for plotting. Defaults to 30.
        filename (str, optional): Filename to save the plot. Defaults to 'plot.png'.

    Returns:
        None
    """
    logging.info(f"Plotting similarities for {filename}")
    sampled_df = df_similarities.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), sample_by_group)))
    logging.info(df_similarities.groupby('label', group_keys=False).mean())
    sns.displot(data=sampled_df, x="similarity", hue='label')
    plt.title('Histogram of similarities between positive examples (1) and negative examples (0)')
    plt.savefig(filename)
