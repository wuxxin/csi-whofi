import os
import glob
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from hampel import hampel

class NTUFiDataset(Dataset):
    """
    PyTorch Dataset class for the NTU-Fi Human Identification (HID) dataset.

    This class loads pre-processed CSI amplitude data from .mat files,
    handles the specific directory structure of the dataset, and prepares
    the data for model training and evaluation.

    Args:
        root_dir (str): The root directory of the dataset (e.g., 'data/NTU-Fi-HumanID').
        split (str): The dataset split to load, either 'train' or 'test'.
        apply_hampel (bool): Whether to apply the Hampel filter for outlier removal.
        augment (bool): Whether to apply data augmentation (only for training split).
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, split='train', apply_hampel=False, augment=False, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.apply_hampel = apply_hampel
        self.augment = augment and self.split == 'train' # Augmentation only for training
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
                mat_contents = loadmat(mat_file)
                data_key = [k for k in mat_contents if isinstance(mat_contents[k], np.ndarray) and k not in ['__header__', '__version__', '__globals__']][0]
                csi_data = mat_contents[data_key].astype(np.float32)

                self.samples.append(csi_data)
                self.labels.append(person_id)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def _apply_augmentations(self, sample):
        """Applies a series of augmentations to a given sample."""
        # Gaussian Noise: N(0, 0.02^2)
        noise = np.random.normal(0, 0.02, sample.shape).astype(np.float32)
        sample += noise

        # Scaling: Scale by a random factor in [0.9, 1.1]
        scaling_factor = np.random.uniform(0.9, 1.1)
        sample *= scaling_factor

        # Time Shift: Shift by a random integer t' in [-5, 5]
        time_shift = np.random.randint(-5, 6)
        if time_shift != 0:
            sample = np.roll(sample, time_shift, axis=0)

        return sample

    def __getitem__(self, idx):
        """
        Retrieves a sample and its label at the given index.
        """
        csi_sample = self.samples[idx].copy()
        label = self.labels[idx]

        # Ensure data is in (antennas, subcarriers, packets) format
        if csi_sample.shape[0] != 3 or csi_sample.shape[1] != 114:
            if csi_sample.shape[1] == 114 and csi_sample.shape[2] == 3:
                csi_sample = np.transpose(csi_sample, (2, 1, 0))
            else:
                raise ValueError(f"Unexpected sample shape: {csi_sample.shape}")

        # Apply Hampel filter if enabled
        if self.apply_hampel:
            # The filter works on 1D data, so we apply it to each subcarrier stream
            for i in range(csi_sample.shape[0]):
                for j in range(csi_sample.shape[1]):
                    res = hampel(csi_sample[i, j, :], window_size=5, n_sigma=3.0)
                    csi_sample[i, j, :] = res.filtered_data

        # Flatten the data: (3, 114, P) -> (P, 342)
        num_packets = csi_sample.shape[2]
        flattened_sample = csi_sample.transpose(2, 0, 1).reshape(num_packets, -1)

        # Apply augmentations if enabled for the training set
        if self.augment:
            flattened_sample = self._apply_augmentations(flattened_sample)

        # Convert to torch tensor
        sample_tensor = torch.from_numpy(flattened_sample).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample_tensor = self.transform(sample_tensor)

        return sample_tensor, label_tensor
