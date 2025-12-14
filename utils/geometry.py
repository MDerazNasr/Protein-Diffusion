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

def dihedral_angles_from_points(p0,p1,p2,p3, eps=1e-8):
    #eps will just prevent division by 0
    '''
    Compute dihedral angles for a sequence of 4 points

    Args:
        p0,p1,p2,p3: Tensors of shape(..., 3)  p0-3 are the 4 points in 3d space
        eps: small constant for numerical stability

    Returns:
        angle: Tensor of shape (...) in radians, range(-pi, pi)    
    '''
    #calc vectors between points
    #In protein language: if points were atoms along the backbone, these are bond directions.
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    #Normalise b1 so it defines the rotation axis cleanly
    #.norm computes vector length (euclidean norm by def)
    #b1_norm is the unit vector length pointing along b1
    #acts as axis of rotation for dihedral angle
    b1_norm = b1/ (torch.linalg.norm(b1, dim=-1, keepdim=True) + eps)

    #Components perpindicular to b1 (project out to axis cleanly)
    #The dihedral is about twisting around b1, so we compare the directions of b0 and b2 after removing any component along b1.
    v = b0 - (b0 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm
    w = b2 - (b2 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm

    #compute angle using atan2 for correct sign
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1_norm, v, dim=-1) * w).sum(dim=-1)

    #torch.atan2(y, x) returns the angle whose tangent is y/x, but it also uses the signs of x and y to put the angle in the correct quadrant.
    angle = torch.atan2(y,x) # (-pi, pi)
    return angle