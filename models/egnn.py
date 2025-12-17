import torch
import torch.nn as nn
import torch.nn.functional as F


class EGNNLayer(nn.Module):
    """
    Minimal EGNN layer.
    Updates node features h and coordinates x using distance-based messages.
    """

    def __init__(self, feat_dim, hidden_dim):
        super().__init__()

        # Edge MLP takes [h_i, h_j, d_ij^2] -> message
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node update: h_i + sum(messages)
        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

        # Coord update scalar: message -> scalar weight for direction
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, x, mask=None):
        """
        h: (B, L, F)
        x: (B, L, 3)
        mask: (B, L) bool
        """
        B, L, Fdim = h.shape

        # -----------------------------
        # 1) Pairwise geometry
        # -----------------------------
        x_i = x[:, :, None, :]                 # (B, L, 1, 3)
        x_j = x[:, None, :, :]                 # (B, 1, L, 3)
        diff = x_i - x_j                       # (B, L, L, 3)
        d2 = (diff ** 2).sum(dim=-1, keepdim=True)  # (B, L, L, 1)

        # Optional: clamp d2 to avoid huge activations early
        d2 = torch.clamp(d2, max=50.0)

        # -----------------------------
        # 2) Edge messages
        # -----------------------------
        h_i = h[:, :, None, :].expand(B, L, L, Fdim)  # (B, L, L, F)
        h_j = h[:, None, :, :].expand(B, L, L, Fdim)  # (B, L, L, F)

        edge_in = torch.cat([h_i, h_j, d2], dim=-1)   # (B, L, L, 2F+1)
        m_ij = self.edge_mlp(edge_in)                 # (B, L, L, H)

        # -----------------------------
        # 3) Mask invalid pairs
        # -----------------------------
        if mask is not None:
            pair_mask = (mask[:, :, None] & mask[:, None, :])       # (B, L, L)
            m_ij = m_ij * pair_mask.unsqueeze(-1).float()

        # -----------------------------
        # 4) Node update
        # -----------------------------
        m_i = m_ij.sum(dim=2)  # sum over neighbors j -> (B, L, H)
        h = h + self.node_mlp(torch.cat([h, m_i], dim=-1))  # (B, L, F)

        # -----------------------------
        # 5) Safe coordinate update
        # -----------------------------
        w_ij = self.coord_mlp(m_ij)            # (B, L, L, 1)
        w_ij = torch.tanh(w_ij) * 0.02         # tiny stable step

        # Remove self edges (diagonal)
        diag = torch.eye(L, device=x.device, dtype=torch.bool).unsqueeze(0)  # (1,L,L)
        w_ij = w_ij.masked_fill(diag.unsqueeze(-1), 0.0)

        # If mask exists, also zero edges touching padded residues
        if mask is not None:
            pair_mask = (mask[:, :, None] & mask[:, None, :])
            w_ij = w_ij * pair_mask.unsqueeze(-1).float()

        # Direction = normalized diff
        dist = torch.sqrt(d2 + 1e-8)           # (B, L, L, 1)
        direction = diff / (dist + 1e-8)       # (B, L, L, 3)

        dx = (w_ij * direction).sum(dim=2)     # (B, L, 3)
        dx = torch.clamp(dx, -0.05, 0.05)      # avoid blowups

        x = x + dx

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # Tripwire for debugging
        if torch.isnan(h).any() or torch.isnan(x).any():
            raise RuntimeError("NaNs detected in EGNNLayer forward()")

        return h, x


