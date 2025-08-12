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
        augment (bool): Whether to apply data augmentation (only for training split).
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, split='train', augment=False, transform=None):
        self.root_dir = root_dir
        self.split = split
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
        The data is loaded as (features, packets) and needs to be
        transposed to (packets, features) for the model.
        """
        csi_sample = self.samples[idx].copy()
        label = self.labels[idx]

        # Data is loaded as (342, P), transpose to (P, 342)
        if csi_sample.shape[0] == 342:
            processed_sample = csi_sample.T
        else:
            raise ValueError(f"Unexpected sample shape: {csi_sample.shape}")

        # Apply augmentations if enabled for the training set
        if self.augment:
            processed_sample = self._apply_augmentations(processed_sample)

        # Convert to torch tensor
        sample_tensor = torch.from_numpy(processed_sample).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample_tensor = self.transform(sample_tensor)

        return sample_tensor, label_tensor
