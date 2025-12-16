"""Diffusion model backbone for protein structure generation."""
import math
import torch
import torch.nn #neural network module
import torch.nn.functional as F #functional versions of operations/layers


#timestep embedding
#denoiser must know whatr noise level its denoising
def sinusodial_timestep_embedding(t, dim=128):
    '''
    Creating sinusodial timestep embeddings like in Transformers

    Args:
        t: Long tensor shape (B, ) timesteps in [0, T-1] --> NOISE LEVEL INDEX, turn t into a vector so NN can condition on it
        dim: embedding dim
    
    Returns:
        emb: Float tensor shape (B, dim) 
    '''
    half = dim // 2 #building half sin and half cos components
    '''
    arrange(start, end) creates [0,1,2,3...half-1]
    dtype= ... ensures float tensor
    device=t.de... puts it on the same device as the t CPU/GPU
    / (half - 1) scales the indices into [0, 1] range-ish. Multiplying by -log(10000) makes values go from 0 down to -log(10000)
    Then torch.exp(...) converts that into frequencies:
        first frequency ≈ 1
        last frequency ≈ 1/10000
        smoothly spaced (log scale)

    why - diff dim oscillate at diff freq so the embeding encodes position/time well
    freq.shape = (half,)
    '''
    freqs = torch.exp( # e^x on a tensor
        -math.log(10000) * torch.arrange(0, half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    # t is (B, ), freqs is (half,) -> (B, half)
    '''
    float - converts int timesteps to float
    unsqueeze - inserts a dim at index 1
              - (B,) becomes (B, 1)
    freqs.unsqueeze(0) - (half,) becomes (1,half)
    so then * to get (B, half)
    so args[b, k] = t[b] * freqs[k]
    '''
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    '''
    torch.sin(args) and torch.cos(args) compute elementwise sine/cosine → each (B, half).
    torch.cat([...], dim=1) concatenates along dimension 1.
    So (B, half) + (B, half) → (B, 2*half) which is (B, dim) when dim is even.
    Why sin+cos:
    gives the model both phase components (richer, stable).
    '''
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1: #If odd, you can’t split evenly into sin/cos halves, so you pad one extra.
        '''
        torch.zeros((B,1)) creates an extra zero column.
        emb.shape[0] is B.
        Concats to make final dimension exactly dim.
        '''
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device)], dim=1)
    return emb

