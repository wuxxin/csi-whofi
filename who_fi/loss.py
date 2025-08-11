import torch
import torch.nn as nn
import torch.nn.functional as F

class InBatchNegativeLoss(nn.Module):
    """
    In-batch negative loss function.
    This loss encourages the model to pull signatures from the same person
    closer together and push signatures from different people further apart.
    It works by using other samples in the batch as negative examples.
    """
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature (float): The temperature scaling factor for the logits.
        """
        super(InBatchNegativeLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, signatures, labels):
        """
        Calculates the in-batch negative loss.

        Args:
            signatures (torch.Tensor): The batch of signatures from the model.
                                       Shape: (batch_size, signature_dim).
            labels (torch.Tensor): The labels for each signature.
                                   Shape: (batch_size,).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # The paper describes a query/gallery setup. A more general and common
        # approach for this kind of loss is to treat every sample in the batch
        # as a query and find its positive/negative pairs within the same batch.

        # Calculate cosine similarity matrix
        # Signatures are assumed to be L2-normalized, so dot product is equivalent
        # to cosine similarity.
        sim_matrix = torch.matmul(signatures, signatures.T)

        # The target for cross-entropy is an identity matrix if all samples
        # in the batch were unique. Here, we have multiple instances of the same
        # person, so we can't directly use this.
        # Instead, we can use a simplified version of NT-Xent loss. For each sample,
        # we treat it as an anchor and the other sample from the same person as
        # the positive sample.

        # This implementation assumes num_instances=2 (query and gallery)
        # This implementation assumes the sampler provides a batch where for each
        # person, there are `num_instances=2`. The batch is structured as:
        # [p1_s1, p1_s2, p2_s1, p2_s2, ...].
        # We need to form query and gallery sets from this structure.

        batch_size, signature_dim = signatures.shape
        num_persons = batch_size // 2

        # Reshape to (num_persons, 2, signature_dim) to separate the two instances
        reshaped_signatures = signatures.view(num_persons, 2, -1)

        # The first instance of each person is the query
        queries = reshaped_signatures[:, 0, :]
        # The second instance of each person is the gallery
        gallery = reshaped_signatures[:, 1, :]

        # Similarity matrix between queries and gallery
        sim_matrix_qg = torch.matmul(queries, gallery.T) / self.temperature

        # The ground truth is that the i-th query matches the i-th gallery item.
        # This corresponds to the diagonal of the similarity matrix.
        # The target labels are therefore [0, 1, 2, ..., N-1].
        targets = torch.arange(num_persons).to(signatures.device)

        # Calculate cross-entropy loss
        loss = self.criterion(sim_matrix_qg, targets)

        return loss
