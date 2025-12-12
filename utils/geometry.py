"""Geometry utilities for protein structures."""

import torch

def pair_wise_distances(x, mask=None):
    '''
    Computer pairwise Euclidean distances between points

    Args:
        x: Tensor of shape (B, L, 3)
            Usually CA atom coordinates
        mask: Optional bool tensor (B, L)
        True for real residues, False for padding

    Returns:
        dist: Tensor of shape (B, L, L)
    '''

    # x[:, :, None, :] -> (B, L, 1, 3)
    # x[:, None, :, :] -> (B, 1, L, 3)
    #subtract every point from every other point
    diff = x[:, :, None, :] - x[:, None, :, :]

    #Square, sum over xyz, then sqrt
    #This uses broadcasting (Pytorch does this automatically)
    #We square differences, sum x/y/z take sqrt
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8) #1e-8 avoids numerical Issues??

    #mask removes padded residues
    if mask is not None:
        #Mask invalid entries (padding)
        mask2d = mask[:, :, None] & mask[:, None, :]
        dist = dist * mask2d

    return dist

def backbone_vectors(x, mask=None):
    """
    Compute backbone direction vectors between consecutive residues

    Args: 
        x: tensor of shape (B, L, 3)
        mask: Optional bool tensor (B, L)
    
    Returns:
        vecs: Tensor of shape (B, L-1, 3)
    """

    #Difference between consecutive residues
    # x[:, 1:] = residue 1 to L
    # x[:, :-1] = residue 0 to L-1
    # Subtract → vector pointing forward along chain
    vecs = x[:, 1:, :] - x[:, :-1, :]

    # Mask removes fake bonds at padding boundaries
    if mask is not None:
        valid = mask[:, 1:] & mask[:, :-1]
        vecs = vecs * valid.unsqueeze(-1)

    return vecs

#normalize vectors (find magnitude ssqrt(x2 + y2))
def normalize_vectors(v, eps=1e-8):
    '''
    Normalise vectors to unit length
    '''
    norm = torch.sqrt((v**2).sum(dim=-1, keepdim=True) + eps)
    return v / norm
