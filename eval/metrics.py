"""Evaluation metrics for protein structures."""

import torch

def masked_mean(x, mask, eps=1e-8):
    '''
    x: (...), mask: same leading shape, bool    
    '''
    mask_f = mask.float()
    return (x * mask_f).sum() / (mask_f.sum().clamp(min=1.0) + eps)

def ca_neighbor_dist(ca_coords, mask):
    """
    Compute CA(i)->CA(i+1) distances.

    ca_coords: (B, L, 3)
    mask:      (B, L) bool

    Returns:
        d: (B, L-1) distances
        d_mask: (B, L-1) valid edges
    """
    diffs = ca_coords[:, 1:, :] - ca_coords[:, :-1, :] #calculate differences between consecutive CAs
    d = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-8) #calculate euclidean distance
    d_mask = mask[:, 1:] & mask[:, :-1] #calculate valid edges
    d = d * d_mask #apply mask to distances
    return d, d_mask

def pairewise_distances(ca_coords, mask):
    """
    Full pairwise distances between CA atoms (mask-aware).

    ca_coords: (B, L, 3)
    mask:      (B, L) bool

    Returns:
        dist: (B, L, L)
        pair_mask: (B, L, L) True where both residues valid
    """   
    diff = ca_coords[:, :, None, :] - ca_coords[:, None, :, :]
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    dist = dist * pair_mask
    return dist, pair_mask

def clash_metrics(ca_coords, mask, ignore_k=3, clash_dist=0.6):
    """
    Clash score for non-neighbor residues.
    
    Think of a protein like a necklace of beads (amino acids). Each bead has a CA atom.
    A "clash" is when two beads that AREN'T next to each other get too close - like 
    two people bumping into each other when they shouldn't be touching!
    
    Biology context:
    - In real proteins, atoms have a minimum safe distance (like personal space)
    - Neighbors (residues i and i+1) are SUPPOSED to be close (~2 Angstroms apart)
    - But non-neighbors should stay far apart (at least 0.6 Angstroms)
    - If non-neighbors get too close, the protein structure is physically impossible
    
    Args:
        ca_coords: (B, L, 3) CA atom coordinates
        mask: (B, L) bool - which residues are real (not padding)
        ignore_k: int - ignore pairs within k positions in sequence (default 3)
                   Why? Because residues 1,2,3 are naturally close in 3D space
                   even if they're far in the sequence!
        clash_dist: float - distance threshold for a clash (default 0.6 Angstroms)
    
    Returns:
        clash_fraction: (B,) - what fraction of non-neighbor pairs are too close
        min_non_neighbor_dist: (B,) - the smallest distance between any non-neighbor pair
    """
    # Step 1: Calculate distances between ALL pairs of CA atoms
    # This creates a big table: dist[i,j] = distance between residue i and residue j
    dist, pair_mask = pairewise_distances(ca_coords, mask)  # (B, L, L)
    B, L, _ = dist.shape

    # Step 2: Build a "non-neighbor mask" - which pairs should we check for clashes?
    # We only care about pairs that are far apart in the sequence (|i-j| > ignore_k)
    # Example: if ignore_k=3, we ignore pairs like (1,2), (1,3), (1,4) but check (1,5), (1,6), etc.
    idx = torch.arange(L, device=ca_coords.device)
    sep = (idx[:, None] - idx[None, :]).abs()  # |i - j| for all pairs
    non_neighbor = sep > ignore_k  # True if residues are far apart in sequence

    # Step 3: Combine masks - we only check pairs that are:
    #   - Both real residues (pair_mask)
    #   - Far apart in sequence (non_neighbor)
    valid = pair_mask & non_neighbor[None, :, :]  # (B, L, L)

    # Step 4: Find the minimum distance between any non-neighbor pair
    # Set invalid pairs to a huge number so they don't affect the minimum
    big = torch.full_like(dist, 1e9)
    dist_valid = torch.where(valid, dist, big)
    min_non_neighbor_dist = dist_valid.min(dim=-1)[0].min(dim=-1)[0]  # (B,)

    # Step 5: Count clashes - how many non-neighbor pairs are too close?
    clashes = (dist < clash_dist) & valid  # True where distance < threshold AND valid

    # Step 6: Calculate the fraction
    # clash_fraction = (number of clashes) / (total valid non-neighbor pairs)
    valid_pairs = valid.sum(dim=(-1, -2)).clamp(min=1).float()  # (B,)
    clash_count = clashes.sum(dim=(-1, -2)).float()  # (B,)
    clash_fraction = clash_count / valid_pairs  # (B,)

    return clash_fraction, min_non_neighbor_dist
