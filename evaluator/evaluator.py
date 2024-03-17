from utils.utils import get_ids_from_query

class ImageNetEvaluator():
    """
    Class to evaluate ImageNet queries.

    Args:
        df (pandas.DataFrame): DataFrame containing the queries and their corresponding results.

    Attributes:
        df (pandas.DataFrame): DataFrame containing the queries and their corresponding results.

    """
    def __init__(self, df):
        self.df = df
    def get_mrr(self,top_k):
        hits = []
        for i, row in self.df.iterrows():
            ranking = get_ids_from_query(row['queries'],top_k)
            try:
                rank_hit = ranking.index(i) + 1
                hits.append(1 / rank_hit)  # Reciprocal of rank
            except ValueError:
                hits.append(0)  # If the relevant document is not in the result list, append 0
        return sum(hits) / len(hits)  # Calculate MRR by taking the mean of reciprocal ranks