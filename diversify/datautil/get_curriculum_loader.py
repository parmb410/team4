# File: diversify/datautil/get_curriculum_loader.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

def get_curriculum_loader(args, algorithm, train_loader):
    """
    A simplified version of how you might compute curriculum‐based subsetting.
    Assume:
      - train_loader yields (x, y, idx) each batch.
      - algorithm.forward(x) returns unnormalized logits (batch_size × num_classes).
    """

    device = next(algorithm.parameters()).device

    # 1) Initialize a per‐sample loss buffer (all zeros to start)
    sample_losses = torch.zeros(len(train_loader.dataset), device=device)

    # 2) Go through one “full pass” to compute losses for every example
    algorithm.eval()
    with torch.no_grad():
        for batch in train_loader:
            x, y, idx = batch                   # unpack triple
            x, y = x.to(device), y.to(device)

            logits = algorithm.forward(x)       # ONLY pass x – no idx!
            # If your code defines a separate `predict` or `inference` method, use that:
            # logits = algorithm.predict(x)

            per_sample_loss = F.cross_entropy(logits, y, reduction='none')
            sample_losses[idx] = per_sample_loss.detach()

    # 3) Compute an ordering or subset based on sample_losses (e.g. sort, EMA, etc.)
    #    Here’s a toy example that picks the “easiest” 50% of samples for this epoch:
    sorted_indices = torch.argsort(sample_losses)  # from lowest loss → highest
    curriculum_size = int(0.5 * len(sorted_indices))

    selected_indices = sorted_indices[:curriculum_size].tolist()
    curriculum_subset = Subset(train_loader.dataset, selected_indices)

    # 4) Create a new DataLoader over just that subset
    curriculum_loader = DataLoader(
        curriculum_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return curriculum_loader
