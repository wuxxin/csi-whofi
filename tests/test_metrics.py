import torch
import pytest
from who_fi.metrics import calculate_metrics

def test_metrics_perfect_ranking():
    """
    Tests the metrics calculation with a perfect ranking.
    The diagonal of the similarity matrix is the highest score.
    """
    num_queries = 10
    num_gallery = 10

    # Perfect similarity matrix (identity matrix)
    sim_matrix = torch.eye(num_queries, num_gallery)
    query_labels = torch.arange(num_queries)
    gallery_labels = torch.arange(num_gallery)

    metrics = calculate_metrics(sim_matrix, query_labels, gallery_labels)

    assert metrics['rank1'] == 1.0
    assert metrics['rank3'] == 1.0
    assert metrics['rank5'] == 1.0
    assert metrics['mAP'] == 1.0

def test_metrics_known_ranking():
    """
    Tests the metrics calculation with a known, imperfect ranking.
    """
    # q0 matches g0 (rank 1)
    # q1 matches g2 (rank 2)
    # q2 matches g1 (rank 3)
    sim_matrix = torch.tensor([
        [0.9, 0.5, 0.2], # Correct is index 0, rank 1
        [0.3, 0.8, 0.9], # Correct is index 2, rank 2 (0.9 is rank 1)
        [0.1, 0.7, 0.4]  # Correct is index 1, rank 3 (0.7 is rank 1, 0.4 is rank 2)
    ])
    query_labels = torch.tensor([0, 2, 1]) # q0 has label 0, q1 has label 2, q2 has label 1
    gallery_labels = torch.tensor([0, 1, 2]) # g0 has label 0, g1 has label 1, g2 has label 2

    # Expected ranks:
    # q0 (label 0) -> finds g0 (label 0) at rank 1
    # q1 (label 2) -> finds g2 (label 2) at rank 2
    # q2 (label 1) -> finds g1 (label 1) at rank 1

    # Let's re-verify the manual calculation.
    # q0 (label 0) wants g0 (label 0). Scores: [0.9, 0.5, 0.2]. Sorted indices: [0, 1, 2]. Rank of g0 is 1. AP = 1/1 = 1.0
    # q1 (label 2) wants g2 (label 2). Scores: [0.3, 0.8, 0.9]. Sorted indices: [2, 1, 0]. Rank of g2 is 1. AP = 1/1 = 1.0
    # q2 (label 1) wants g1 (label 1). Scores: [0.1, 0.7, 0.4]. Sorted indices: [1, 2, 0]. Rank of g1 is 1. AP = 1/1 = 1.0

    # The labels need to be aligned with the similarity matrix.
    # sim_matrix[i,j] is sim between query i and gallery j.
    # query_labels[i] should match gallery_labels[j] for a positive pair.
    # Let's make it simpler.

    sim_matrix = torch.tensor([
        [0.9, 0.1, 0.2], # q0 -> g0, rank 1
        [0.1, 0.3, 0.8], # q1 -> g2, rank 1
        [0.2, 0.7, 0.3]  # q2 -> g1, rank 1
    ])
    query_labels = torch.tensor([10, 20, 30])
    gallery_labels = torch.tensor([10, 30, 20])

    # q0 (label 10) matches g0 (label 10). Rank 1.
    # q1 (label 20) matches g2 (label 20). Rank 1.
    # q2 (label 30) matches g1 (label 30). Rank 1.
    # This is another perfect ranking case.

    # Let's try again with a mixed-rank case.
    sim_matrix = torch.tensor([
        [0.9, 0.1, 0.2], # q0 (label 10) -> g0 (label 10). Rank 1. AP=1/1
        [0.1, 0.8, 0.3], # q1 (label 20) -> g2 (label 20). find g2. scores for q1 are [0.1, 0.8, 0.3]. sorted indices: [1, 2, 0]. g2 is at index 2. rank is 2. AP = 1/2
        [0.7, 0.3, 0.8]  # q2 (label 30) -> g1 (label 30). find g1. scores for q2 are [0.7, 0.3, 0.8]. sorted indices: [2, 0, 1]. g1 is at index 1. rank is 3. AP = 1/3
    ])
    query_labels = torch.tensor([10, 20, 30])
    gallery_labels = torch.tensor([10, 30, 20]) # g0=10, g1=30, g2=20

    # Expected ranks: 1, 2, 3
    # Rank-1 hits: 1
    # Rank-3 hits: 3
    # Rank-5 hits: 3
    # mAP = (1/1 + 1/2 + 1/3) / 3 = (1.833) / 3 = 0.611

    metrics = calculate_metrics(sim_matrix, query_labels, gallery_labels)

    assert metrics['rank1'] == pytest.approx(1/3)
    assert metrics['rank3'] == pytest.approx(3/3)
    assert metrics['rank5'] == pytest.approx(3/3)
    assert metrics['mAP'] == pytest.approx((1/1 + 1/2 + 1/3) / 3)
