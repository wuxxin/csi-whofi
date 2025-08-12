from who_fi.data import NTUFiDataset

print("Inspecting training dataset...")
train_dataset = NTUFiDataset(root_dir='data/NTU-Fi-HumanID', split='train')
print(f"Number of samples: {len(train_dataset)}")
num_persons = len(set(train_dataset.labels))
print(f"Number of unique persons: {num_persons}")

print("\nInspecting test dataset...")
test_dataset = NTUFiDataset(root_dir='data/NTU-Fi-HumanID', split='test')
print(f"Number of samples: {len(test_dataset)}")
num_persons = len(set(test_dataset.labels))
print(f"Number of unique persons: {num_persons}")
