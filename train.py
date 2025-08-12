import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from who_fi.data import NTUFiDataset
from who_fi.sampler import InBatchSampler
from who_fi.model import WhoFiTransformer
from who_fi.loss import InBatchNegativeLoss
from who_fi.metrics import calculate_metrics

def main(args):
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Loading ---
    print("Loading datasets...")
    train_dataset = NTUFiDataset(
        root_dir=args.data_dir,
        split='train',
        augment=args.augment
    )

    test_dataset = NTUFiDataset(
        root_dir=args.data_dir,
        split='test'
    )

    train_sampler = InBatchSampler(
        data_source=train_dataset,
        batch_size=args.batch_size,
        num_instances=2 # For query-gallery pairs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, # For evaluation, we can use a standard batch size
        shuffle=False,
        num_workers=args.num_workers
    )

    # --- 3. Model, Loss, Optimizer ---
    print("Initializing model, loss, and optimizer...")
    model = WhoFiTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        signature_dim=args.signature_dim
    ).to(device)

    loss_fn = InBatchNegativeLoss(temperature=args.temperature).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )

    # --- 4. Training & Evaluation Loop ---
    print("Starting training...")
    best_rank1 = 0.0

    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")

        # Train
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            metrics = evaluate(model, test_dataset, device, args.batch_size)
            print(f"Epoch {epoch+1} Evaluation Metrics: "
                  f"Rank-1: {metrics['rank1']:.4f}, "
                  f"Rank-3: {metrics['rank3']:.4f}, "
                  f"Rank-5: {metrics['rank5']:.4f}, "
                  f"mAP: {metrics['mAP']:.4f}")

            # Save best model
            if metrics['rank1'] > best_rank1:
                best_rank1 = metrics['rank1']
                print(f"New best Rank-1: {best_rank1:.4f}. Saving model...")
                torch.save(model.state_dict(), 'best_model.pth')

        # Step the scheduler
        scheduler.step()

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for i, (samples, labels) in enumerate(train_loader):
        samples = samples.to(device)
        # Labels are not used in the loss function directly, but are used by the sampler

        optimizer.zero_grad()

        signatures = model(samples)

        loss = loss_fn(signatures, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)

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
    # For each person, one sample is gallery, rest are queries
    unique_labels = torch.unique(all_labels)
    query_signatures, query_labels = [], []
    gallery_signatures, gallery_labels = [], []

    for pid in unique_labels:
        person_indices = (all_labels == pid).nonzero(as_tuple=True)[0]
        gallery_idx = person_indices[0] # First sample as gallery
        query_indices = person_indices[1:] # Rest as queries

        gallery_signatures.append(all_signatures[gallery_idx])
        gallery_labels.append(all_labels[gallery_idx])

        if len(query_indices) > 0:
            query_signatures.append(all_signatures[query_indices])
            query_labels.append(all_labels[query_indices])

    query_signatures = torch.cat(query_signatures, dim=0)
    query_labels = torch.cat(query_labels, dim=0)
    gallery_signatures = torch.stack(gallery_signatures, dim=0)
    gallery_labels = torch.stack(gallery_labels, dim=0)

    # Compute similarity matrix
    sim_matrix = torch.matmul(query_signatures, gallery_signatures.T)

    # Calculate metrics
    metrics = calculate_metrics(sim_matrix, query_labels, gallery_labels)

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the WhoFi Transformer model.")

    # Data and environment
    parser.add_argument('--data_dir', type=str, default='data/NTU-Fi-HumanID', help='Directory for the dataset')
    parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA training')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')

    # Training hyperparameters
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (P*K, e.g., 4 persons * 2 instances)')
    parser.add_argument('--eval_interval', type=int, default=10, help='Run evaluation every N epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--lr_step_size', type=int, default=50, help='Step size for LR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.95, help='Gamma for LR scheduler')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for loss function')

    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in the transformer')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of transformer encoder layers')
    parser.add_argument('--signature_dim', type=int, default=128, help='Dimension of the final signature')

    args = parser.parse_args()
    main(args)
