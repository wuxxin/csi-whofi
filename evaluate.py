import argparse
import torch
from torch.utils.data import DataLoader

from who_fi.data import NTUFiDataset
from who_fi.model import WhoFiTransformer
from who_fi.metrics import calculate_metrics

def evaluate(model, test_dataset, device, batch_size):
    model.eval()

    # Extract all features from the test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_signatures = []
    all_labels = []

    with torch.no_grad():
        for samples, labels in test_loader:
            samples = samples.to(device)
            signatures = model(samples)
            all_signatures.append(signatures.cpu())
            all_labels.append(labels.cpu())

    all_signatures = torch.cat(all_signatures, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Create query and gallery sets
    unique_labels = torch.unique(all_labels)
    query_signatures, query_labels = [], []
    gallery_signatures, gallery_labels = [], []

    for pid in unique_labels:
        person_indices = (all_labels == pid).nonzero(as_tuple=True)[0]
        gallery_idx = person_indices[0]
        query_indices = person_indices[1:]

        gallery_signatures.append(all_signatures[gallery_idx])
        gallery_labels.append(all_labels[gallery_idx])

        if len(query_indices) > 0:
            query_signatures.append(all_signatures[query_indices])
            query_labels.append(all_labels[query_indices])

    query_signatures = torch.cat(query_signatures, dim=0)
    query_labels = torch.cat(query_labels, dim=0)
    gallery_signatures = torch.stack(gallery_signatures, dim=0)
    gallery_labels = torch.stack(gallery_labels, dim=0)

    sim_matrix = torch.matmul(query_signatures, gallery_signatures.T)

    metrics = calculate_metrics(sim_matrix, query_labels, gallery_labels)

    return metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    print("Loading test dataset...")
    test_dataset = NTUFiDataset(
        root_dir=args.data_dir,
        split='test',
        augment=False
    )

    print("Initializing model...")
    model = WhoFiTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        signature_dim=args.signature_dim
    ).to(device)

    print(f"Loading model weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    print("Running evaluation...")
    metrics = evaluate(model, test_dataset, device, args.batch_size)

    print("\n--- Evaluation Results ---")
    print(f"Rank-1: {metrics['rank1']:.4f}")
    print(f"Rank-3: {metrics['rank3']:.4f}")
    print(f"Rank-5: {metrics['rank5']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    print("--------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the WhoFi Transformer model.")

    # Config
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/NTU-Fi-HumanID', help='Directory for the dataset')
    parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction')

    # Model hyperparameters (must match the trained model)
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in the transformer')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of transformer encoder layers')
    parser.add_argument('--signature_dim', type=int, default=128, help='Dimension of the final signature')

    args = parser.parse_args()
    main(args)
