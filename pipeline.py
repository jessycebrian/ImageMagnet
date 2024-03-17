import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets.load_data import load_data
from datasets.clipdataset import CLIPDataset
from pinecone_connection import right_index
from utils import model, processor,device
from data2vdb import CLIP2Vdb  
from utils.similarities import plot_similarities
import seaborn as sns
import matplotlib.pyplot as plt

from evaluator.evaluator import ImageNetEvaluator

logging.info('Initialising ImageMagnet...')
# load data and transform
df = load_data(debug=5) # Remove debug if entire dataset to be processed.
# create clipdataset
clip_dataset = CLIPDataset(df)
# initialise CLIP2VDB to upload vectors
clip2Vdb = CLIP2Vdb(model = model, processor=processor, device=device, vdb_index = right_index)
# generate embeddings
clip2Vdb.get_embeddings(clip_dataset)
# upload to vectordb
clip2Vdb.upsert_embeddings(df)

# Get similarities for embeddings.
# df_similarities_t = clip2Vdb.measure_embeddings_similarities(clip2Vdb.text_embeddings)
# df_similarities_c = clip2Vdb.measure_embeddings_similarities(clip2Vdb.final_embeddings)
df_similarities_i = clip2Vdb.measure_embeddings_similarities(clip2Vdb.image_embeddings)

# plot_similarities(df_similarities_t, filename = 'plot_text.png')
plot_similarities(df_similarities_i, filename = 'plot_image.png')
# plot_similarities(df_similarities_c, filename = 'plot_combined.png')

logging.info('Evaluating ImageMagnet ...')
# initialise evaluator
evaluator = ImageNetEvaluator(df)
top_k=5
logging.info(f"MRR@{top_k}is {evaluator.get_mrr(top_k = top_k)}")


