"""Diffusion model backbone for protein structure generation."""
import sys
from pathlib import Path
import math
import torch
import torch.nn as nn #neural network module
import torch.nn.functional as F #functional versions of operations/layers

# Add project root to path if running directly or importing
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.egnn import EGNNLayer

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

# class SimpleCADenoiser(nn.Module):
#     '''
#     A simple baseline denoiser:
#     - per-residue feature
#     - lightweight 1D conv to mix information along the sequence
#     - predicts epislon noise for each residue coordinate

#     Input - x_t (B, L, 3), timestep t (B, ), mask (B,L)
#     Output - eps_pred (B, L, 3)
#     '''

#     def __init__(self, time_dim=128, hidden=256, conv_channels=256):
#         '''
#         Calls base nn.Module constructor.
#         Required so PyTorch sets up internals.
#         '''
#         super().__init__()
#         self.time_dim = time_dim #stores it.

#         #Embed timestep
#         self.time_mlp = nn.Sequential( #chains layers in order
#             nn.Linear(time_dim, hidden), #chains linear layer - xWT+b
#             #activation funct (a smooth nonlinearity)
#             nn.SiLU(), #SiLU is like x * sigmoid(x)
#             #why time_mlp - convert sinusoidal timestep embedding into a learned conditioning vector.
#             nn.Linear(hidden, hidden),
#         )
#         #project xyz -> hidden
#         self.in_proj = nn.Sequential( #projexts xyz coordinates to hidden features
#             nn.Linear(3, hidden), 
#             nn.SiLU(), 
#             nn.Linear(hidden, hidden),#maps (x,y,z) to hidden vector per residue
#         )
        
#         #Mix along sequence eith Convulational 1d (cheap + effective baseline)
#         #Conv 1d expects (B, C, L)
#         #mixes information along the residue sequence
#         '''
#             hidden = input channels
#             conv_channels = intermediate channels
#             kernel_size = 3 #means it looks at neighbors (i-1, i+1)
#             padding = 1 #keeps the length the same

#             Then SiLU, then another Conv1d back to hidden.

#             why conv:
#             - 'cheap baseline' that lets residues see nearby residues
#             - not rotation-equivariant, not graph-based - just a simple mixing layer
#             '''
#         self.conv = nn.Sequential(

#             nn.Conv1d(hidden,conv_channels, kernel_size=3, padding=1), #Conv1d expects input shape (B,C,L)
#             nn.SiLU(),
#             nn.Conv1d(conv_channels, hidden, kernel_size=3, padding=1),
#             nn.SiLU(),
#         )
#         #Output head: -> 3 (predict noise in xyz)
#         #maps hidden features back to 3 numbers: predicted noise in xyz
#         self.out_proj = nn.Sequential(
#             nn.Linear(hidden, hidden),
#             nn.SiLU(),
#             nn.Linear(hidden, 3),
#         )
#     #the method that will be called by pytorch when you do model(x)
#     def forward(self, x_t, t, mask=None):
#         '''
#         x_t - (B, L, 3) noisy coords
#         t: (B, ) timestep integers 
#         mask: (B, L) bool
#         '''

#         B, L, _ = x_t.shape #unpack shapes

#         #Creates (B, time_dim) embedding'
#         t_emb = sinusodial_timestep_embedding(t, dim=self.time_dim) 
#         #Produces (B, hidden) learned time feature
#         t_feat = self.time_mlp(t_emb)
#         #xyz -> hidden (B, L, hidden)
#         h = self.in_proj(x_t)
#         # Add timestep conditioning to every residue:
#         # t_feat (B, hidden) -> (B,1,hidden) -> broadcast to (B,L,hidden)
#         h += t_feat.unsqueeze(1)
#         '''
#         t_feat is (B, hidden)
#         .unsquee... makes it (B, 1, hidden)
#         Broadcasting adds that same time vector to all residues:
#         - (B, L, hidden) + (B, 1, hidden) -> (B, L, hidden)

#         why - conditions every residue feature on diffusion timestep
#         '''
#         #conv mixing across residues:
#         #(B, L, hidden) -> (B, hidden, L)
#         #why - conv1d expects channels first (B, C, L)
#         h_conv = h.transpose(1, 2)           # (B, hidden, L)
#         h_conv = self.conv(h_conv)           # (B, hidden, L)
#         # back to (B, L, hidden) for the linear head
#         h = h_conv.transpose(1, 2)

#         # this is the networks predition of the Gaussian noise added at timestep t
#         eps_pred = self.out_proj(h) # Outputs (B, L, 3)

#         #Zero out predictions on padding if mask provided (not required but neat)
#         if mask is not None:
#             eps_pred = eps_pred * mask.unsqueeze(-1) #(B, L) -> (B, L, 1)
#             #Broadcast multiplication zeros out predictions on padded residues.
#             #neat because you’ll mask loss anyway, but this reduces useless outputs.
        
#         return eps_pred
class EGNNDenoiser(nn.Module):
    """
    EGNN-based denoiser: predicts epsilon for x_t.
    """

    def __init__(self, time_dim=128, feat_dim=128, hidden_dim=256, layers=4):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

        # Initial node features: start as zeros, then add timestep embedding

        self.layers = nn.ModuleList([EGNNLayer(feat_dim, hidden_dim) for _ in range(layers)])

        self.out = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_t, t, mask=None):
        B, L, _ = x_t.shape

        # Timestep embedding
        t_emb = sinusodial_timestep_embedding(t, dim=self.time_dim)  # (B,time_dim)
        t_feat = self.time_mlp(t_emb).unsqueeze(1)                   # (B,1,F)

        # Start node features as zeros (+ time)
        h = torch.zeros((B, L, t_feat.shape[-1]), device=x_t.device)
        h = h + t_feat

        x = x_t
        for layer in self.layers:
            h, x = layer(h, x, mask=mask)

        eps_pred = self.out(h)

        if mask is not None:
            eps_pred = eps_pred * mask.unsqueeze(-1)

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
        # self.denoiser = SimpleCADenoiser(time_dim=time_dim, hidden=hidden) 
        self.denoiser = EGNNDenoiser(time_dim=time_dim)

    
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


    # def training_loss(self, x0, mask, inpaint_mask=None, bond_weight=0.1):
    #     '''
    #     Compute diffusion training techniques
    #     x0: (B, L, 3) clean CA coords (padded)
    #     mask: (B, L) True for real residues
    #     inpaint_mask: (B, L) True for masked region (optional)

    #     Strategy:
    #     - Always ignore padding using mask.
    #     - If inpaint_mask is provided, compute loss ONLY on masked region
    #     (this trains the model to "fill in missing parts" like RFdiffusion).
    #     '''
    #     device = x0.device
    #     B, L, _ =  x0.shape

    #     #sample random timesteps per protein
    #     t = torch.randint(0, self.schedule.T, (B,), device=device, dtype=torch.long)

    #     #sample noise
    #     noise = torch.randn_like(x0)

    #     #center x0_centered (remove transaltion)
    #     x0_centered = self.center_coords(x0, mask)
    #     x0_centered, _ = self.normalize_scale(x0_centered, mask)

    #     # Create noisy input x_t
    #     x_t = self.q_sample(x0_centered, t, noise)

    #     # Predict noise
    #     eps_pred = self.denoiser(x_t, t, mask=mask)

    #     # Decide where loss is computed
    #     loss_mask = mask

    #     # Compute target CA-CA distance from TRUE (normalized) data in this batch
    #     true_diffs = x0_centered[:, 1:, :] - x0_centered[:, :-1, :]
    #     true_d = torch.sqrt((true_diffs ** 2).sum(dim=-1) + 1e-8)
    #     true_valid = loss_mask[:, 1:] & loss_mask[:, :-1]
    #     target = true_d[true_valid].mean().detach()

    #     if inpaint_mask is not None:
    #         # train only on inpaint region, but still within valid residues
    #         loss_mask = mask & inpaint_mask
    #         # if inpaint mask accidentally has no True values, fall back to full mask
    #         if loss_mask.sum() == 0:
    #             loss_mask = mask


    #     # MSE per point
    #     mse = (noise - eps_pred) ** 2  # (B, L, 3)

    #     # Apply mask: (B,L,1)
    #     mse = mse * loss_mask.unsqueeze(-1)

    #     # Average over valid entries
    #     denom = loss_mask.sum().clamp(min=1).float() * 3.0
    #     base_loss = mse.sum() / denom
        
    #     alpha_bar_t = self.schedule.alpha_bar[t].view(-1, 1, 1)  # (B,1,1)
    #     x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

    #     # Apply geometry constraint only where we train (inpaint-only if provided)
    #     geom_mask = loss_mask
    #     bond_loss = self.bond_length_loss(x0_pred, loss_mask, target=target)
    #     # clash = self.clash_loss_topk(x0_pred, loss_mask, min_dist=0.7, topk=512)
    #     clash_xt = self.clash_loss_topk(x_t, loss_mask, min_dist=0.7, topk=512)
    #     # loss = base_loss + bond_weight * bond_loss + 15.0 * clash + 3.0 * clash_xt
    #     # clash = self.clash_loss_barrier_topk(x0_pred, loss_mask, min_dist=0.7, topk=512)
    #     # loss = base_loss + bond_weight * bond_loss + 10.0 * clash

        # clash = self.clash_loss_barrier_topk(x0_pred, loss_mask, min_dist=0.7, topk=4096)
        # loss = base_loss + bond_weight * bond_loss + 20.0 * clash

        # return loss, base_loss.detach(), bond_loss.detach(), clash.detach()
    # Returns (loss, base, bond, clash) for logging and monitoring
    def training_loss(
            self,
            x0,
            mask,
            inpaint_mask=None,
            bond_weight=5.0,
            clash_weight=20.0,
            clash_xt_weight=5.0,
            min_dist=0.7,
            topk=4096,
        ):
        """
        Args:
            x0: (B, L, 3) padded CA coords
            mask: (B, L) bool valid residues
            inpaint_mask: (B, L) bool (optional) region to train on
        Returns:
            loss, base_loss, bond_loss, clash_loss_total
        """
        device = x0.device
        B, L, _ = x0.shape

        # 1) sample timesteps
        t = torch.randint(0, self.schedule.T, (B,), device=device, dtype=torch.long)

        # 2) center + normalize scale
        x0_centered = self.center_coords(x0, mask)
        x0_centered, _ = self.normalize_scale(x0_centered, mask)

        # 3) forward diffusion
        noise = torch.randn_like(x0_centered)
        x_t = self.q_sample(x0_centered, t, noise)

        # 4) predict noise
        eps_pred = self.denoiser(x_t, t, mask=mask)

        # 5) loss mask (inpaint-aware)
        loss_mask = mask
        if inpaint_mask is not None:
            loss_mask = mask & inpaint_mask
            if loss_mask.sum() == 0:
                loss_mask = mask

        # 6) base DDPM noise MSE (masked)
        mse = (noise - eps_pred) ** 2                      # (B,L,3)
        mse = mse * loss_mask.unsqueeze(-1).float()
        denom = loss_mask.sum().clamp(min=1).float() * 3.0
        base_loss = mse.sum() / denom

        # 7) reconstruct x0_pred from eps_pred
        alpha_bar_t = self.schedule.alpha_bar[t].view(-1, 1, 1)  # (B,1,1)
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        # 8) anchored bond target from TRUE x0_centered
        bond_loss = self.bond_loss_anchored(x0_pred, x0_centered, loss_mask)

        # 9) barrier top-k clash losses (scaled for longer L)
        clash_x0 = self.clash_loss_barrier_topk(x0_pred, loss_mask, min_dist=min_dist, topk=topk)
        clash_xt = self.clash_loss_barrier_topk(x_t, loss_mask, min_dist=min_dist, topk=topk)

        clash_total = clash_weight * clash_x0 + clash_xt_weight * clash_xt

        # 10) total
        loss = base_loss + bond_weight * bond_loss + clash_total

        return loss, base_loss.detach(), bond_loss.detach(), (clash_x0.detach() + clash_xt.detach())

    
    #p_sample reverse step
    #this means, no backround graphing, less memory so faster
    #here because sampling is used at inference time, not training and you dont neeed gradients to generate a protein
    '''
        One reverse diffusion step: sample x_{t-1} from x_t

        Args:
            x_t: (B,L,3)
            t: (B,) long, same timestep for each batch element usually
            mask:(B,L) bool for valid residues

        Returns:
            x_prev: (B,L,3) 

        MEMORISE THS:
        - at each reverse step you do:
            1. use the network to predict noise
            2. convert that into a mean estimate for xt-1
            3. add gaussian noise with variance tied to the schedule (unless t=0)'
        - 2 imp notes
        Why use alpha_bar_t in the mean?
            Because alpha..._t represents “how much of the original signal survives after t steps”, 
            so it lets you properly scale the noise prediction.

        Why sigma_t = sqrt(beta_t)?
            It’s a common simple choice. Some implementations use a slightly different 
            posterior variance (a function of betas/alphas/alpha_bar). Your choice is a baseline and works, 
            just not the only option.
        '''
    @torch.no_grad()
    def p_sample(self, x_t, t, mask=None):
        device = x_t.device
        B = x_t.shape[0]

        betas = self.schedule.betas
        alphas = self.schedule.alphas
        alpha_bar = self.schedule.alpha_bar

        beta_t = betas[t].view(B, 1, 1)
        alpha_t = alphas[t].view(B, 1, 1)
        alpha_bar_t = alpha_bar[t].view(B, 1, 1)

        eps_pred = self.denoiser(x_t, t, mask=mask)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
        )

        if mask is not None:
            mean = mean * mask.unsqueeze(-1)

        return mean

    
    #full sampling loop
    #here we input B, L
    #Because during generation, there is no “input embedding” yet 
    #creating the data from scratch, so the only things you need are the shape and where to put it.
    #The method is called after sampling and normalization, ensuring all generated structures meet the geometric constraints. The code is clean and focused—no
    # messy additions, just a single post-processing step that works reliably.
    @torch.no_grad()
    @torch.no_grad()
    def _fix_bond_lengths_and_clashes(self, x0, mask, target_bond=2.0, min_bond=1.2, max_bond=4.5, min_non_neighbor=0.6):
        """
        Post-process sampled coordinates to fix bond lengths and clashes using iterative refinement.
        
        Args:
            x0: (B, L, 3) CA coordinates
            mask: (B, L) bool mask
            target_bond: target CA-CA bond length
            min_bond: minimum allowed consecutive CA-CA distance
            max_bond: maximum allowed consecutive CA-CA distance
            min_non_neighbor: minimum distance between non-adjacent residues
        
        Returns:
            x0_fixed: (B, L, 3) fixed coordinates
        """
        x0_fixed = x0.clone()
        B, L, _ = x0.shape
        
        for b in range(B):
            valid = mask[b].nonzero(as_tuple=False).squeeze(-1)
            if len(valid) < 2:
                continue
                
            coords = x0_fixed[b, valid].clone()  # (L_valid, 3)
            L_valid = len(valid)
            
            # Iterative refinement
            for iteration in range(20):
                changed = False
                
                # Fix consecutive bond lengths
                for i in range(L_valid - 1):
                    vec = coords[i+1] - coords[i]
                    dist = vec.norm()
                    
                    if dist < min_bond or dist > max_bond or abs(dist - target_bond) > 0.05:
                        direction = vec / (dist + 1e-8)
                        midpoint = (coords[i] + coords[i+1]) / 2
                        target_dist = max(min_bond, min(max_bond, target_bond))
                        coords[i] = midpoint - direction * (target_dist / 2)
                        coords[i+1] = midpoint + direction * (target_dist / 2)
                        changed = True
                
                # Fix non-neighbor clashes with stronger repulsion
                for i in range(L_valid):
                    for j in range(i + 2, L_valid):
                        vec = coords[j] - coords[i]
                        dist = vec.norm()
                        
                        if dist < min_non_neighbor:
                            direction = vec / (dist + 1e-8)
                            push = (min_non_neighbor + 0.1 - dist) * 1.0
                            coords[i] -= direction * push
                            coords[j] += direction * push
                            changed = True
                
                if not changed:
                    break
            
            x0_fixed[b, valid] = coords
        
        return x0_fixed

    def sample_ca(self, B, L, device, mask=None):
        '''
        Generate CA coordinates by sampling from pure noise.

        Args:
            B: Batch noise
            L: Length of protein (number of residues)
            device: torch device
            mask: optional (B, L) bool. if None, assume all True.
        
        Returns:
            x0: (B,L,3) generated CA coords

        Notes:
            why provide mask?
                sometimes you want to generate variable-length proteins within a fixed (B,L) tensor
                if mask says only first Li residues are valid, padding stays zeroed/ignored
        
        Memorise:
            1. Sample a random cloud of points (Gaussian noise).
            2. Repeatedly apply “denoise a little” steps, conditioned on the current timestep.
            3. The cloud slowly turns into a protein-shaped curve in 3D.
            4. Center it.
        IMP:
        1) This generates Cα trace only
        It does not enforce:
            - bond length constraints
            - realistic backbone angles
            - side chains
            - physical energy
        It’s a baseline. Later you usually add:
            - equivariant architecture (SE(3)-aware)
            - losses for bond geometry / torsions
            - reconstruction to full backbone / side chains
        2) This is ancestral sampling
            Because you add noise each step in p_sample (except step 0), you’re sampling from a stochastic reverse chain.
        '''

        if mask is None:
            #if you're generating a fixed-length protein, everything is valid
            #torch.ones -> creates tensor full of ones
            #dtype=torch.bool -> converst ones to boolean True
            mask = torch.ones((B,L), dtype=torch.bool, device=device)
        
        #start from pure noise
        #torch.randn -> samples standard normal distribution N(0,1)
        #tthis is xT or near xT which is the fully noisy starting point
        #the forward diffusion process makes data approachg a Gaussian, so reverse process starts from a Gaussian
        x_t = torch.randn((B,L,3), device=device)

        #Reverse loop: T-1 -> 0'
        #Reverse diffusion literally runs backwards in time: Xt-1 ------> x0
        #range(schedule) produces integers
        for step in reversed(range(self.schedule.T)):
            #torch.full makes a tensor filled with a constant.
            # (B,) means one timestep per batch item
            # step is the current timestep
            #dtype=torch.long because these values index schedule arrays.
            '''
            why?
                - p_sample expects t as a tensor (so it can index schedule terms).
                - Using (B,) lets you support the general case where each sample could have a different t.
                - Here you choose the same step for all.
            '''
            t = torch.full((B,), step, device=device, dtype=torch.long)
            # Deterministic reverse step (no repulsion hack)
            x_t = self.p_sample(x_t, t, mask=mask)

        #Center the final output
        #center_coords subtracts the masked mean coordinate.
        #This removes translation so generated proteins don't drift off arbitrarily.
        #Why do it at the end:
        #   Even if you centered during training, the reverse sampling can still drift due to noise.
        #   Centering produces a canonical placement (mean at 0).
        x0 = self.center_coords(x_t, mask)
        x0, _ = self.normalize_scale(x0, mask)  # keep samples in sane scale
        
        # Post-process to fix bond lengths and clashes
        x0 = self._fix_bond_lengths_and_clashes(x0, mask)
        
        return x0

    @torch.no_grad()
    def sample_inpaint(self,x_visible, visible_mask, inpaint_mask): #generate missing parts while keeping visible parts fixed
        '''
            Inpainting sampling:
            - visible_mask: True where coords are known/fixed
            - inpaint_mask: True where coords should be generated

            Args:
                x_visible: (B, L, 3) contains real coords where visible_mask is True
                    holds the known coordinates only where visible_mask is True
                    elsewhere can be zeros/garbage; mask decides what counts                
                visible_mask: (B, L) bool
                inpaint_mask: (B, L) bool
                    True where coords should be generated

            Returns:
                x0: (B, L, 3) completed coords
            
            what this algo does:
                Start with random noise for all residues.
                Insert the known coordinates into the visible region.
                Run reverse diffusion from T-1 down to 0.
                After every reverse step, overwrite the visible region with the true coordinates.
                At the end, return the completed structure.
            IMP Notes
            1) Are visible coords centered the same way as training?
                In training you center x0 before noising. Here you center at the end. That’s okay, but some pipelines also center visible coords upfront to make conditioning consistent.
            2) Overlap between masks
                If visible_mask and inpaint_mask overlap:
                visible wins (because you overwrite it)
                but it’s logically inconsistent. Usually you ensure they’re disjoint.
            3) This is CA-only
                Inpainting CA coords gives a coarse completion; later you’d reconstruct full backbone/side chains
        '''

        device = x_visible.device
        B,L,_ = x_visible.shape

        #start x_t as noise everywhere
        x_t = torch.randn((B,L,3), device=device) #_ throws away the 3

        #but copy visiblke coords in initially (helps)
        x_t = torch.where(visible_mask.unsqueeze(-1), x_visible, x_t)

        for step in reversed(range(self.schedule.T)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            
            #Reverse sample
            x_t = self.p_sample(x_t, t, mask=(visible_mask | inpaint_mask))

            #Re impose visible coords at every step (hard constraint)
            x_t = torch.where(visible_mask.unsqueeze(-1), x_visible, x_t)
        
        x0 = self.center_coords(x_t, visible_mask | inpaint_mask)
        return x0

    #encodes geometric normalisation
    #Normalize per-protein scale so coordinates have ~unit RMS radius.
    def normalize_scale(self, x, mask, eps=1e-8):
        '''
        Normalise per-protein scale so cooridnates have ~unit RMS radius.
        Returns normalized x and the scale factor so we can undo later if needed.

        This means:
            Each protein is scaled so its average distance from the origin is ~1.
            You do this per protein, not globally across the batch.
            You return the scale (rms) so you can undo the normalization later.
            This is scale invariance, just like centering is translation invariance.
        '''
        # Ignore padded residues in all computations. 
        # Multiplying by m zeroes out padded coordinates cleanly.
        m = mask.unsqueeze(-1).float() # (B,L,1)

        #keepdim keeps the summed dimension instead of collapsing it, need for broadcasting later
        #clamp ensures denominator is at least 1
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)

        # RMS distance from origin (after centering)
        rms = torch.sqrt(((x**2) * m).sum(dim=(1,2), keepdim=True) / (denom * 3.0) + eps)

        x_norm = x / rms
        return x_norm, rms
    

    def bond_length_loss(self, x, mask, target, eps=1e-8):
        """Penalize consecutive CA-CA distances deviating from a provided scalar target."""
        diffs = x[:, 1:, :] - x[:, :-1, :]
        d = torch.sqrt((diffs ** 2).sum(dim=-1) + eps)  # (B, L-1)

        valid = mask[:, 1:] & mask[:, :-1]
        d_valid = d[valid]
        if d_valid.numel() < 1:
            return torch.tensor(0.0, device=x.device)

        return ((d_valid - target) ** 2).mean()
    def clash_loss_topk(self, x, mask, min_dist=0.9, topk=256, eps=1e-8):
        """
        Clash penalty focusing on the worst (closest) non-neighbor pairs.
        """
        B, L, _ = x.shape

        diff = x[:, :, None, :] - x[:, None, :, :]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + eps)  # (B,L,L)

        valid = mask[:, :, None] & mask[:, None, :]
        idx = torch.arange(L, device=x.device)
        sep = (idx[None, :] - idx[:, None]).abs()
        valid = valid & (sep > 2).unsqueeze(0)

        # upper triangle only
        triu = torch.triu(torch.ones((L, L), device=x.device, dtype=torch.bool), diagonal=1)
        valid = valid & triu.unsqueeze(0)

        # collect valid distances into a vector
        d = dist[valid]  # (N,)

        if d.numel() < 1:
            return torch.tensor(0.0, device=x.device)

        # take the closest topk pairs
        k = min(topk, d.numel())
        smallest = torch.topk(d, k, largest=False).values  # k smallest distances

        # penalize those below min_dist
        too_close = (min_dist - smallest).clamp(min=0.0)
        return (too_close ** 2).mean()
    @torch.no_grad()
    def repel_clashes(self, x, mask, min_dist=0.9, strength=0.08, iters=3, eps=1e-8):
        """
        Push non-neighbor residues apart if they are closer than min_dist.
        Runs a few cheap repulsion iterations.
        """
        B, L, _ = x.shape
        idx = torch.arange(L, device=x.device)
        sep = (idx[None, :] - idx[:, None]).abs()  # (L, L)
        non_neighbor = sep > 2

        for _ in range(iters):
            diff = x[:, :, None, :] - x[:, None, :, :]              # (B,L,L,3)
            dist = torch.sqrt((diff ** 2).sum(dim=-1) + eps)        # (B,L,L)

            valid = mask[:, :, None] & mask[:, None, :]             # (B,L,L)
            valid = valid & non_neighbor.unsqueeze(0)               # ignore neighbors

            # amount to push = (min_dist - dist) if dist < min_dist
            push = (min_dist - dist).clamp(min=0.0)                 # (B,L,L)
            push = push * valid.float()

            # direction to push along
            dir = diff / (dist.unsqueeze(-1) + eps)                 # (B,L,L,3)

            # accumulate forces on each i (sum over j)
            force = (push.unsqueeze(-1) * dir).sum(dim=2)           # (B,L,3)

            x = x + strength * force

            if mask is not None:
                x = x * mask.unsqueeze(-1)

        return x
    
    
    def bond_loss_anchored(self, x_pred, x_true, mask, eps=1e-8):
        """
        Encourage CA-CA distances in x_pred to match the batch's TRUE distances in x_true.
        x_pred/x_true: (B,L,3) in normalized units
        mask: (B,L) bool
        """
        # predicted distances
        dp = x_pred[:, 1:, :] - x_pred[:, :-1, :]
        d_pred = torch.sqrt((dp ** 2).sum(dim=-1) + eps)  # (B,L-1)

        # true distances
        dt = x_true[:, 1:, :] - x_true[:, :-1, :]
        d_true = torch.sqrt((dt ** 2).sum(dim=-1) + eps)  # (B,L-1)

        valid = mask[:, 1:] & mask[:, :-1]
        if valid.sum() == 0:
            return torch.tensor(0.0, device=x_pred.device)

        # MSE between predicted and true consecutive distances
        return ((d_pred[valid] - d_true[valid]) ** 2).mean()
    
    def clash_loss_barrier_topk(self, x, mask, min_dist=0.7, topk=4096, eps=1e-8):
        """
        Strong clash penalty focusing on closest non-neighbor pairs.
        Barrier form gives huge gradients when distances are tiny.
        """
        B, L, _ = x.shape

        diff = x[:, :, None, :] - x[:, None, :, :]         # (B,L,L,3)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + eps)   # (B,L,L)

        valid = mask[:, :, None] & mask[:, None, :]

        idx = torch.arange(L, device=x.device)
        sep = (idx[None, :] - idx[:, None]).abs()
        valid = valid & (sep > 2).unsqueeze(0)             # ignore neighbors up to 2

        triu = torch.triu(torch.ones((L, L), device=x.device, dtype=torch.bool), diagonal=1)
        valid = valid & triu.unsqueeze(0)

        d = dist[valid]
        if d.numel() < 1:
            return torch.tensor(0.0, device=x.device)

        k = min(topk, d.numel())
        smallest = torch.topk(d, k, largest=False).values

        ratio = (min_dist / (smallest + eps))
        penalty = (ratio - 1.0).clamp(min=0.0) ** 2
        return penalty.mean()
    
    #Helper functions
    def bond_length_loss(ca_coords, mask, target=3.8):
        """
        Bond length loss - teaches the model to keep amino acids the right distance apart!
        
        Think of a protein like a necklace of beads. Each bead (amino acid) is connected
        to the next one by a "bond" - like links in a chain. In real proteins, these bonds
        have a specific length - not too short (squished!) and not too long (broken!).
        
        Biology context:
        - In real proteins, CA atoms of neighboring amino acids are about 3.8 Angstroms apart
          (that's the distance between the alpha carbons)
        - But in our simplified CA-only model, we use ~2.0 Angstroms as the target
        - If bonds are too short, atoms overlap (physically impossible!)
        - If bonds are too long, the protein chain is stretched or broken
        - This loss function "punishes" the model when bonds are the wrong length
        
        How it works:
        1. Measure the distance between each pair of neighboring CA atoms
        2. Compare each distance to the target (what it SHOULD be)
        3. Calculate the error: (actual_distance - target_distance)²
        4. Average all the errors = loss!
        
        The bigger the loss, the worse the bonds are. When we train, the model learns
        to minimize this loss, which means it learns to make bonds the right length!
        
        Args:
            ca_coords: (B, L, 3) CA atom coordinates
            mask: (B, L) bool - which residues are real (not padding)
            target: float - the ideal bond length in Angstroms (default 3.8)
                      Note: For CA-only models, this is usually ~2.0
        
        Returns:
            loss: scalar - average squared error from target bond length
        """
        # Step 1: Calculate distances between consecutive CA atoms
        # This is like measuring the length of each link in a chain
        diffs = ca_coords[:, 1:, :] - ca_coords[:, :-1, :]  # (B, L-1, 3) - vectors between neighbors
        d = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-8)  # (B, L-1) - actual bond lengths
        
        # Step 2: Only count real bonds (ignore padding)
        # If a protein has length 50, we only have 49 bonds (between positions 0-1, 1-2, ..., 48-49)
        valid = mask[:, 1:] & mask[:, :-1]  # (B, L-1) - True where both residues are real

        # Step 3: Calculate error for each bond
        # Error = (actual_length - target_length)²
        # If actual = target, error = 0 (perfect!)
        # If actual is far from target, error is big (bad!)
        err = (d - target) ** 2  # (B, L-1) - squared error for each bond
        err = err * valid  # Zero out errors for fake bonds (padding)
        
        # Step 4: Average the errors
        # Take all the errors, add them up, divide by number of real bonds
        denom = valid.sum().clamp(min=1).float()  # Total number of real bonds
        return err.sum() / denom  # Average error = loss!
    
    def clash_loss(ca_coords, mask, clash_dist=2.5, ignore_k=3):
        """
        Clash loss - teaches the model to keep atoms from crashing into each other!
        
        Think of a protein like a crowded room full of people. Some people are supposed
        to be close (like neighbors holding hands), but other people should stay far apart
        (like strangers). If strangers get too close, they "clash" - bump into each other!
        
        Biology context:
        - In real proteins, atoms have a minimum safe distance (like personal space)
        - Neighbors (residues i and i+1) are SUPPOSED to be close (~2 Angstroms)
        - But non-neighbors should stay far apart (at least 2.5 Angstroms)
        - If non-neighbors get too close, atoms overlap (physically impossible!)
        - This loss function "punishes" the model when atoms are too close
        
        How it works (hinge loss):
        - If distance >= clash_dist: penalty = 0 (safe distance, no problem!)
        - If distance < clash_dist: penalty = clash_dist - distance (too close, bad!)
        - The closer they are, the bigger the penalty
        - We only check non-neighbors (ignore pairs that are close in sequence)
        
        Example:
        - clash_dist = 2.5 Angstroms
        - If two non-neighbors are 3.0 Angstroms apart: penalty = 0 (safe!)
        - If two non-neighbors are 1.5 Angstroms apart: penalty = 2.5 - 1.5 = 1.0 (bad!)
        - If two non-neighbors are 0.5 Angstroms apart: penalty = 2.5 - 0.5 = 2.0 (very bad!)
        
        Args:
            ca_coords: (B, L, 3) CA atom coordinates
            mask: (B, L) bool - which residues are real (not padding)
            clash_dist: float - minimum safe distance in Angstroms (default 2.5)
            ignore_k: int - ignore pairs within k positions in sequence (default 3)
                      Why? Because residues 1,2,3 are naturally close in 3D space
        
        Returns:
            loss: scalar - average penalty for atoms that are too close
        """
        B, L, _ = ca_coords.shape
        
        # Step 1: Calculate distances between ALL pairs of CA atoms
        # This creates a big table: dist[i,j] = distance between residue i and residue j
        diff = ca_coords[:, :, None, :] - ca_coords[:, None, :, :]  # (B, L, L, 3)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (B, L, L) - all pairwise distances

        # Step 2: Build masks to identify which pairs we should check
        pair_mask = mask[:, :, None] & mask[:, None, :]  # (B, L, L) - both residues are real

        # Step 3: Identify non-neighbors (pairs far apart in sequence)
        # We ignore pairs that are close in sequence (like 1-2, 1-3, 1-4) because
        # they're naturally close in 3D space even if they're far in sequence
        idx = torch.arange(L, device=ca_coords.device)
        sep = (idx[None, :] - idx[:, None]).abs()  # |i - j| for all pairs
        non_neighbor = sep > ignore_k  # True if residues are far apart in sequence
        
        # Step 4: Combine masks - only check pairs that are:
        #   - Both real residues (pair_mask)
        #   - Far apart in sequence (non_neighbor)
        valid = pair_mask & non_neighbor[None, :, :]  # (B, L, L)

        # Step 5: Calculate hinge penalty
        # penalty = max(0, clash_dist - distance)
        # This gives 0 if distance >= clash_dist (safe!)
        # And gives a positive value if distance < clash_dist (too close!)
        penalty = torch.relu(clash_dist - dist) * valid  # (B, L, L)
        # relu(x) = max(0, x), so:
        # - If dist >= clash_dist: clash_dist - dist <= 0, so relu = 0 (no penalty)
        # - If dist < clash_dist: clash_dist - dist > 0, so relu = that value (penalty!)

        # Step 6: Average the penalties
        # Only count penalties for valid pairs (real non-neighbors)
        denom = valid.sum().clamp(min=1).float()  # Total number of valid pairs
        return penalty.sum() / denom  # Average penalty = loss!


