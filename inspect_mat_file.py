import glob
from scipy.io import loadmat
import numpy as np

# Find one .mat file to inspect
mat_file = glob.glob('data/NTU-Fi-HumanID/train_amp/*/*.mat')[0]
print(f"Inspecting file: {mat_file}")

mat_contents = loadmat(mat_file)
data_key = [k for k in mat_contents if isinstance(mat_contents[k], np.ndarray) and k not in ['__header__', '__version__', '__globals__']][0]
csi_data = mat_contents[data_key]

print(f"Data key: {data_key}")
print(f"Data shape: {csi_data.shape}")
