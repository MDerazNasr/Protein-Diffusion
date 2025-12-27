"""Evaluation metrics for protein structures."""

import torch

def masked_mean(x, mask, eps=1e-8):
    '''
    x: (...), mask: same leading shape, bool    
    '''
    mask_f = mask.float()
    return (x * mask_f).sum() / (mask_f.sum() + eps)
