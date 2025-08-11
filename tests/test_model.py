import torch
import pytest
from who_fi.model import WhoFiTransformer

@pytest.fixture
def model():
    """Returns a WhoFiTransformer instance with default parameters."""
    return WhoFiTransformer()

@pytest.fixture
def sample_input():
    """Returns a sample input tensor for the model."""
    batch_size = 4
    seq_len = 100 # Number of packets
    input_dim = 342
    return torch.randn(batch_size, seq_len, input_dim)

def test_model_instantiation(model):
    """
    Tests if the WhoFiTransformer model can be instantiated correctly.
    """
    assert isinstance(model, WhoFiTransformer)

def test_model_forward_pass_shape(model, sample_input):
    """
    Tests if the forward pass of the model returns an output with the expected shape.
    """
    signature_dim = 128
    output = model(sample_input)

    assert output.shape[0] == sample_input.shape[0] # Batch size should be the same
    assert output.shape[1] == signature_dim # Output dimension should be signature_dim
    assert len(output.shape) == 2

def test_model_output_normalization(model, sample_input):
    """
    Tests if the output signatures are L2-normalized.
    The norm of each signature vector should be close to 1.
    """
    output = model(sample_input)
    norms = torch.norm(output, p=2, dim=1)

    # Check if all norms are close to 1
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
