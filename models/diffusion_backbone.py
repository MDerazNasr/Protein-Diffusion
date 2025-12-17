"""Diffusion model backbone for protein structure generation."""
import math
import torch
import torch.nn as nn #neural network module
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
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / (half - 1)
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

#Noise schedule
def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    '''
    Linear schedule from beta-start to beta_end over T steps.
    Returns betas: (T, )

    T - number of diffusion steps
    beta_start, beta_end = noise variance range
    device="cpu" default
    '''

    #torch.linspace(a,b,T) makes T evenly spaced values from a to b
    #returns betas shape (T, )

    return torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float32)

class DiffusionSchedule:
    '''
    Docstring says:
        alpha_t = 1 - beta_t
        alpha_bar_t = ∏_{s=0..t} alpha_s
        Those are the standard DDPM terms.

    stores diffusion schedule terms for DDPM:
        beta_t, alpha_t = 1-beta_t, alpha_bar_t = prod alpha_0..t
    '''

    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T
        betas = linear_beta_schedule(T, beta_start, beta_end, device=device) #builds beta tensor
        alphas = 1.0 - betas #subtract element wise
        '''
        cumprod - cumulative product
        dim=0 coz its a 1D tensor
        alpha_bar[t] = alphas[0]*alphas[1]*...*alphas[t].
        '''
        alpha_bar = torch.cumprod(alphas, dim=0) 

        self.betas = betas #(T, )
        self.alphas = alphas #(T, )
        self.alpha_bar = alpha_bar #(T, )

    #convienience method to move schedule tensors to GPU/CPU
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

#simple denoiser network
'''
Inherits from nn.Module:
gives you parameter registration
.to(device), .parameters(), training/eval modes, etc.
'''
#This network predicts epsilon noise (the DDPM objective)
# - input: noisy coords x_t and timestep t
# - output: predicted noise ε̂

class SimpleCADenoiser(nn.Module):
    '''
    A simple baseline denoiser:
    - per-residue feature
    - lightweight 1D conv to mix information along the sequence
    - predicts epislon noise for each residue coordinate

    Input - x_t (B, L, 3), timestep t (B, ), mask (B,L)
    Output - eps_pred (B, L, 3)
    '''

    def __init__(self, time_dim=128, hidden=256, conv_channels=256):
        '''
        Calls base nn.Module constructor.
        Required so PyTorch sets up internals.
        '''
        super().__init__()
        self.time_dim = time_dim #stores it.

        #Embed timestep
        self.time_mlp = nn.Sequential( #chains layers in order
            nn.Linear(time_dim, hidden), #chains linear layer - xWT+b
            #activation funct (a smooth nonlinearity)
            nn.SiLU(), #SiLU is like x * sigmoid(x)
            #why time_mlp - convert sinusoidal timestep embedding into a learned conditioning vector.
            nn.Linear(hidden, hidden),
        )
        #project xyz -> hidden
        self.in_proj = nn.Sequential( #projexts xyz coordinates to hidden features
            nn.Linear(3, hidden), 
            nn.SiLU(), 
            nn.Linear(hidden, hidden),#maps (x,y,z) to hidden vector per residue
        )
        
        #Mix along sequence eith Convulational 1d (cheap + effective baseline)
        #Conv 1d expects (B, C, L)
        #mixes information along the residue sequence
        '''
            hidden = input channels
            conv_channels = intermediate channels
            kernel_size = 3 #means it looks at neighbors (i-1, i+1)
            padding = 1 #keeps the length the same

            Then SiLU, then another Conv1d back to hidden.

            why conv:
            - 'cheap baseline' that lets residues see nearby residues
            - not rotation-equivariant, not graph-based - just a simple mixing layer
            '''
        self.conv = nn.Sequential(

            nn.Conv1d(hidden,conv_channels, kernel_size=3, padding=1), #Conv1d expects input shape (B,C,L)
            nn.SiLU(),
            nn.Conv1d(conv_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        #Output head: -> 3 (predict noise in xyz)
        #maps hidden features back to 3 numbers: predicted noise in xyz
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
    #the method that will be called by pytorch when you do model(x)
    def forward(self, x_t, t, mask=None):
        '''
        x_t - (B, L, 3) noisy coords
        t: (B, ) timestep integers 
        mask: (B, L) bool
        '''

        B, L, _ = x_t.shape #unpack shapes

        #Creates (B, time_dim) embedding'
        t_emb = sinusodial_timestep_embedding(t, dim=self.time_dim) 
        #Produces (B, hidden) learned time feature
        t_feat = self.time_mlp(t_emb)
        #xyz -> hidden (B, L, hidden)
        h = self.in_proj(x_t)
        # Add timestep conditioning to every residue:
        # t_feat (B, hidden) -> (B,1,hidden) -> broadcast to (B,L,hidden)
        h += t_feat.unsqueeze(1)
        '''
        t_feat is (B, hidden)
        .unsquee... makes it (B, 1, hidden)
        Broadcasting adds that same time vector to all residues:
        - (B, L, hidden) + (B, 1, hidden) -> (B, L, hidden)

        why - conditions every residue feature on diffusion timestep
        '''
        #conv mixing across residues:
        #(B, L, hidden) -> (B, hidden, L)
        #why - conv1d expects channels first (B, C, L)
        h_conv = h.transpose(1, 2)           # (B, hidden, L)
        h_conv = self.conv(h_conv)           # (B, hidden, L)
        # back to (B, L, hidden) for the linear head
        h = h_conv.transpose(1, 2)

        # this is the networks predition of the Gaussian noise added at timestep t
        eps_pred = self.out_proj(h) # Outputs (B, L, 3)

        #Zero out predictions on padding if mask provided (not required but neat)
        if mask is not None:
            eps_pred = eps_pred * mask.unsqueeze(-1) #(B, L) -> (B, L, 1)
            #Broadcast multiplication zeros out predictions on padded residues.
            #neat because you’ll mask loss anyway, but this reduces useless outputs.
        
        return eps_pred
    
# Diffusion model wrapper
class BackboneDiffusionModel(nn.Module):
    '''
    Wraps a denoiser + schedule, implements:
    - forward diffusion: q(x_t | x_0)
    - training loss: MSE between true noise and predicted noise

    wraps:
    - schedule (betas/alphas)
    - denoiser network
    - forward diffusion sampling
    - training loss
    '''
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, time_dim=12, hidden=256):
        super().__init__() #initialise module
        self.schedule = DiffusionSchedule(T=T, beta_start=beta_start, beta_end=beta_end) 
        self.denoiser = SimpleCADenoiser(time_dim=time_dim, hidden=hidden) 
    
    def to(self, device):
        super().to(device)
        self.schedule.to(device)
        return self
    
    def center_coords(self, x, mask):
        '''
        Remove translation by centering coordinates per protein.
        This helps because proteins can be anywhere in space.
        proteins are only meaningful up to rigid transformations
        centering eliminates 'where it is in space' as a nuisance factor
        '''
        #mask: (B,L) bool
        '''
        mask (B, L) → (B, L, 1).
        .float() converts bool → float (True=1.0, False=0.0).
        m becomes a weighting tensor.
        '''
        m = mask.unsqueeze(-1).float() #(B,L,1)
        #sum - counts how many valid residues
        #keepdim = True keeps (B,1,1)
        #.clamp(min=1.0) prevents division by zero, if somehow a protein had 0 valid residues
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0) #(B,1,1)
        #x * m zeros padded residues
        #sum over residues gives (B,1,3)
        #divide by denom gives masked mean coordinate
        mean = (x * m).sum(dim=1, keepdim=True) / denom
        return x - mean #mean becomes around 0
    
    def q_sample(self, x0, t, noise):
        """
        Forward diffusion: sample x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise

        x0: (B, L, 3)
        t:  (B,) long
        noise: (B, L, 3)

        """
        #gather alpha_bar[t] for each batch element
        alpha_bar_t = self.schedule.alpha_bar[t].view(-1,1,1) #(B,1,1)
        '''
        self.schedule.alpha_bar is shape (T,).
        Indexing with t (shape (B,)) gives alpha_bar_t shape (B,).
        .view(-1, 1, 1) reshapes to (B,1,1) so it broadcasts across (B,L,3).
        '''
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        '''
        torch.sqrt elementwise.
            Multiplication broadcasts:
            (B,1,1) times (B,L,3) → (B,L,3).
            Interpretation:
            at small t: alpha_bar close to 1 → mostly x0
            at large t: alpha_bar small → mostly noise
        '''


    def training_loss(self, x0, mask, inpaint_mask=None):
        '''
        Compute diffusion training techniques
        x0: (B, L, 3) clean CA coords (padded)
        mask: (B, L) True for real residues
        inpaint_mask: (B, L) True for masked region (optional)

        Strategy:
        - Always ignore padding using mask.
        - If inpaint_mask is provided, compute loss ONLY on masked region
        (this trains the model to "fill in missing parts" like RFdiffusion).
        '''
        device = x0.device
        B, L, _ =  x0.shape

        #sample random timesteps per protein
        t = torch.randint(0, self.schedule.T, (B,), device=device, dtype=torch.long)

        #sample noise
        noise = torch.randn_like(x0)

        #center x0_centered (remove transaltion)
        x0_centered = self.center_coords(x0, mask)

        # Create noisy input x_t
        x_t = self.q_sample(x0_centered, t, noise)

        # Predict noise
        eps_pred = self.denoiser(x_t, t, mask=mask)

        # Decide where loss is computed
        loss_mask = mask
        if inpaint_mask is not None:
            # train only on inpaint region, but still within valid residues
            loss_mask = mask & inpaint_mask
            # if inpaint mask accidentally has no True values, fall back to full mask
            if loss_mask.sum() == 0:
                loss_mask = mask

        # MSE per point
        mse = (noise - eps_pred) ** 2  # (B, L, 3)

        # Apply mask: (B,L,1)
        mse = mse * loss_mask.unsqueeze(-1)

        # Average over valid entries
        denom = loss_mask.sum().clamp(min=1).float() * 3.0
        loss = mse.sum() / denom

        return loss


                