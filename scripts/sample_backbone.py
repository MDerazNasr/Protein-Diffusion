import os
import torch
import sys
from pathlib import Path
import json

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.diffusion_backbone import BackboneDiffusionModel
from utils.pdb_io import write_ca_pdb
from eval.metrics import compute_backbone_metrics



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "checkpoints/backbone_diffusion_epoch5.pt"  # change if needed
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    T = ckpt.get("T", 500)

    model = BackboneDiffusionModel(T=T).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Sample a protein backbone
    B = 1
    L = 80  # try 46 first if you only trained on 46, then increase
    ca = model.sample_ca(B=B, L=L, device=device)  # (1, L, 3)

    out_path = os.path.join(out_dir, f"sample_L{L}.pdb")
    write_ca_pdb(ca[0], out_path)

    print(f"Saved sample to: {out_path}")

    print("CA min/max:", ca.min().item(), ca.max().item())
    print("CA mean norm:", ca.norm(dim=-1).mean().item())

    #Build a mask (all residues are valid here)
    mask = torch.ones((1,L), dtype=torch.bool, device=device)
    metrics = compute_backbone_metrics(ca, mask)
    print("Sample metrics:", metrics)

    metrics_path = os.path.join(out_dir, f"sample_L{L}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Saved metrics to: {metrics_path}")

    d = (ca[:, 1:, :] - ca[:, :-1, :]).norm(dim=-1)  # (B, L-1)
    print("CA-CA distance mean:", d.mean().item())
    print("CA-CA distance min/max:", d.min().item(), d.max().item())
    print("Any NaNs in coords:", torch.isnan(ca).any().item())

    x = ca[0]  # (L,3)
    diff = x[:, None, :] - x[None, :, :]
    dist = (diff**2).sum(dim=-1).sqrt()

    L = x.shape[0]
    idx = torch.arange(L)
    sep = (idx[None,:] - idx[:,None]).abs()
    dist = dist[sep > 2]  # non-neighbors only

    print("Min non-neighbor distance:", dist.min().item())



if __name__ == "__main__":
    main()


