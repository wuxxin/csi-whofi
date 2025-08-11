import os
import glob
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

class NTUFiDataset(Dataset):
    """
    PyTorch Dataset class for the NTU-Fi Human Identification (HID) dataset.

    This class loads pre-processed CSI amplitude data from .mat files,
    handles the specific directory structure of the dataset, and prepares
    the data for model training and evaluation.

    Args:
        root_dir (str): The root directory of the dataset (e.g., 'data/NTU-Fi-HumanID').
        split (str): The dataset split to load, either 'train' or 'test'.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.labels = []

        if self.split == 'train':
            self.data_path = os.path.join(self.root_dir, 'train_amp')
        elif self.split == 'test':
            self.data_path = os.path.join(self.root_dir, 'test_amp')
        else:
            raise ValueError("split must be 'train' or 'test'")

        self._load_data()

    def _load_data(self):
        """
        Loads data from .mat files into memory.
        The directory structure is expected to be:
        <root_dir>/<split_amp>/<person_id>/<sample_file>.mat
        """
        person_id_folders = glob.glob(os.path.join(self.data_path, '*'))

        for person_folder in person_id_folders:
            person_id = int(os.path.basename(person_folder))
            mat_files = glob.glob(os.path.join(person_folder, '*.mat'))

            for mat_file in mat_files:
                # The actual data is stored under a key, usually 'whitened_csi' or similar.
                # We need to find the key that contains the data array.
                mat_contents = loadmat(mat_file)
                data_key = [k for k in mat_contents if isinstance(mat_contents[k], np.ndarray) and k not in ['__header__', '__version__', '__globals__']][0]
                csi_data = mat_contents[data_key]

                self.samples.append(csi_data)
                self.labels.append(person_id)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a sample and its label at the given index.

        The method performs the required data flattening:
        - Input shape: (3, 114, P) where P is the number of packets.
        - Output shape: (P, 342)

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: (sample, label) where sample is the flattened CSI data
                   and label is the integer person ID.
        """
        csi_sample = self.samples[idx]
        label = self.labels[idx]

        # Ensure data is in (antennas, subcarriers, packets) format
        # The raw data is often (packets, subcarriers, antennas) or similar,
        # so we need to inspect and permute if necessary.
        # Assuming the loaded data is (3, 114, P) as per the paper.
        if csi_sample.shape[0] != 3 or csi_sample.shape[1] != 114:
             # Let's assume it might be (P, 114, 3) and permute
            if csi_sample.shape[1] == 114 and csi_sample.shape[2] == 3:
                csi_sample = np.transpose(csi_sample, (2, 1, 0)) # (3, 114, P)
            else:
                # If shapes are unexpected, we might need to debug here.
                # For now, we raise an error.
                raise ValueError(f"Unexpected sample shape: {csi_sample.shape}")

        # Flatten the data: (3, 114, P) -> (P, 3*114) = (P, 342)
        num_packets = csi_sample.shape[2]
        flattened_sample = csi_sample.transpose(2, 0, 1).reshape(num_packets, -1)

        # Convert to torch tensor
        sample_tensor = torch.from_numpy(flattened_sample).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample_tensor = self.transform(sample_tensor)

        return sample_tensor, label_tensor
