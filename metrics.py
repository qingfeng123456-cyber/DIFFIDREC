import numpy as np


def dcg(relevance_scores):
    """Compute the Discounted Cumulative Gain."""
    relevance_scores = np.asfarray(relevance_scores)  # Ensure scores is an array of floats
    return np.sum((2**relevance_scores - 1) / np.log2(np.arange(2, relevance_scores.size + 2)))

def ndcg_at_k(relevance_scores, top_k):
    """Compute NDCG at rank k."""
    relevance_scores = np.asfarray(relevance_scores)[:top_k]  # Ensure r is an array of floats and take top k scores
    dcg_max = dcg(sorted(relevance_scores, reverse=True))
    if not dcg_max:
        return 0.
    return dcg(relevance_scores) / dcg_max
