import torch
import pytest
import torch.nn.functional as F
from who_fi.loss import InBatchNegativeLoss

@pytest.fixture
def loss_fn():
    """Returns an InBatchNegativeLoss instance."""
    return InBatchNegativeLoss(temperature=0.1)

def test_loss_perfect_prediction(loss_fn):
    """
    Tests the loss function with perfect predictions.
    If positive pairs are identical, the loss should be minimal.
    """
    num_persons = 4
    batch_size = num_persons * 2
    signature_dim = 128

    # Create signatures where positive pairs are identical
    base_signatures = F.normalize(torch.randn(num_persons, signature_dim), p=2, dim=1)

    # Batch is [s1, s1, s2, s2, ...]
    signatures = base_signatures.repeat_interleave(2, dim=0)

    # Labels are not used by this loss implementation, but required by the function signature
    labels = torch.arange(num_persons).repeat_interleave(2)

    loss = loss_fn(signatures, labels)

    # Loss should be low. The exact value is -log(exp(1/T) / sum(exp(sim/T))).
    # For T=0.1, perfect match sim=1, others are smaller.
    # The similarity of a query with its positive gallery is 1.
    # The similarity with other gallery items is smaller.
    # The loss should be small.
    assert loss.item() < 1.0

def test_loss_worst_case_prediction(loss_fn):
    """
    Tests the loss function with a worst-case scenario where a negative
    pair is a perfect match and the positive pair is orthogonal.
    """
    num_persons = 4
    batch_size = num_persons * 2
    signature_dim = 128

    queries = F.normalize(torch.randn(num_persons, signature_dim), p=2, dim=1)
    # Create orthogonal gallery samples for positive pairs
    gallery = F.normalize(torch.randn(num_persons, signature_dim), p=2, dim=1)

    # Make a negative pair be a perfect match
    # q0 should match g1, not g0.
    gallery[1] = queries[0].clone()

    # Interleave to form the batch [q0, g0, q1, g1, ...]
    signatures = torch.stack([queries, gallery], dim=1).view(batch_size, -1)

    labels = torch.arange(num_persons).repeat_interleave(2)

    loss = loss_fn(signatures, labels)

    # We expect a high loss value because the highest similarity is for a negative pair.
    assert loss.item() > 2.0
