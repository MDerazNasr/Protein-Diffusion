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

    