import pytest
from who_fi.sampler import InBatchSampler
from collections import namedtuple

# Mock dataset for testing
MockDataset = namedtuple('MockDataset', ['labels'])

def test_sampler_batch_size():
    """Tests if the sampler produces batches of the correct size."""
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    dataset = MockDataset(labels=labels)
    batch_size = 8
    num_instances = 2
    sampler = InBatchSampler(dataset, batch_size, num_instances)

    batches = list(sampler)
    # 4 persons total, 4 persons per batch = 1 batch
    assert len(batches) == 1
    for batch in batches:
        assert len(batch) == batch_size

def test_sampler_content():
    """Tests if each batch has the correct number of instances per person."""
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    dataset = MockDataset(labels=labels)
    batch_size = 6
    num_instances = 3
    sampler = InBatchSampler(dataset, batch_size, num_instances)

    num_persons_per_batch = batch_size // num_instances

    for batch_indices in sampler:
        batch_labels = [labels[i] for i in batch_indices]
        label_counts = {label: batch_labels.count(label) for label in set(batch_labels)}

        assert len(set(batch_labels)) == num_persons_per_batch
        for label, count in label_counts.items():
            assert count == num_instances

def test_sampler_drop_last():
    """Tests if the sampler drops the last batch if it's not full."""
    # 5 persons, but batch size requires 4 persons per batch (8/2)
    labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    dataset = MockDataset(labels=labels)
    batch_size = 8
    num_instances = 2
    sampler = InBatchSampler(dataset, batch_size, num_instances)

    # Expect only one full batch
    assert len(list(sampler)) == 1

def test_sampler_invalid_args():
    """Tests if the sampler raises an error for invalid arguments."""
    labels = [0, 0, 1, 1]
    dataset = MockDataset(labels=labels)
    with pytest.raises(ValueError):
        # batch_size not divisible by num_instances
        InBatchSampler(dataset, batch_size=3, num_instances=2)
