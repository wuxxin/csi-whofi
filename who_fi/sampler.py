import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np
from typing import Iterator

class InBatchSampler(Sampler[list[int]]):
    """
    Custom sampler to create batches for in-batch negative loss.
    Each batch contains P persons, with K instances each.
    This is a common sampling strategy for person re-identification.

    Args:
        data_source: The dataset to sample from.
        batch_size (int): The total number of samples in a batch.
        num_instances (int): The number of instances to sample for each person (K).
    """
    def __init__(self, data_source, batch_size: int, num_instances: int):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances

        if self.batch_size % self.num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")

        self.num_persons_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        # Assuming data_source has a 'labels' attribute
        for index, label in enumerate(self.data_source.labels):
            self.index_dic[label].append(index)

        self.pids = list(self.index_dic.keys())

        # Filter out persons with fewer than num_instances samples
        self.pids = [pid for pid in self.pids if len(self.index_dic[pid]) >= self.num_instances]

        self.num_samples = len(self.pids)

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle the list of person IDs
        np.random.shuffle(self.pids)

        # Create batches
        for i in range(0, self.num_samples, self.num_persons_per_batch):
            batch_pids = self.pids[i : i + self.num_persons_per_batch]

            if len(batch_pids) < self.num_persons_per_batch:
                # Drop the last batch if it's not full
                continue

            batch_indices = []
            for pid in batch_pids:
                # For each person, randomly sample num_instances
                indices = np.random.choice(
                    self.index_dic[pid], self.num_instances, replace=False
                )
                batch_indices.extend(indices)

            yield batch_indices

    def __len__(self) -> int:
        return self.num_samples // self.num_persons_per_batch
