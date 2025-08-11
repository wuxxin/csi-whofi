import torch
import numpy as np

def calculate_metrics(sim_matrix, query_labels, gallery_labels):
    """
    Calculates re-identification metrics (Rank-k accuracy and mAP).

    Args:
        sim_matrix (torch.Tensor): A similarity matrix of shape
                                   (num_queries, num_gallery).
        query_labels (torch.Tensor): Labels for the query samples.
        gallery_labels (torch.Tensor): Labels for the gallery samples.

    Returns:
        dict: A dictionary containing the calculated metrics:
              'rank1', 'rank3', 'rank5', 'mAP'.
    """
    num_queries, num_gallery = sim_matrix.shape

    # Sort gallery samples for each query based on similarity
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)

    # Match matrix: 1 if query and gallery labels match, 0 otherwise
    matches = (query_labels.unsqueeze(1) == gallery_labels[sorted_indices]).float()

    # Calculate Rank-k accuracy
    rank1 = 0
    rank3 = 0
    rank5 = 0
    aps = []

    for i in range(num_queries):
        query_matches = matches[i]

        # Find the rank of the first correct match
        # The rank is the index of the first '1' in the matches vector
        rank = torch.nonzero(query_matches).min().item() + 1 # 1-indexed rank

        if rank <= 1:
            rank1 += 1
        if rank <= 3:
            rank3 += 1
        if rank <= 5:
            rank5 += 1

        # Calculate Average Precision (AP) for this query
        # Since there is only one ground truth for each query, AP is 1/rank
        ap = 1.0 / rank
        aps.append(ap)

    rank1_acc = rank1 / num_queries
    rank3_acc = rank3 / num_queries
    rank5_acc = rank5 / num_queries
    mAP = np.mean(aps)

    return {
        'rank1': rank1_acc,
        'rank3': rank3_acc,
        'rank5': rank5_acc,
        'mAP': mAP,
    }
