import torch
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import create_dataloader

loader = create_dataloader("data/processed", batch_size=1)

batch = next(iter(loader))

print("coords:", batch["coords"].shape)
print("ca_coords:", batch["ca_coords"].shape)
print("pairwise_dist:", batch["pairwise_dist"].shape)
print("bond_vecs:", batch["bond_vecs"].shape)

print("torsion_angles:", batch["torsion_angles"].shape)
print("torsion_sincos:", batch["torsion_sincos"].shape)
print("torsion_mask:", batch["torsion_mask"].shape)

print("mask sum:", batch["mask"].sum().item())
print("inpaint sum:", batch["inpaint_mask"].sum().item())
print("visible sum:", batch["visible_mask"].sum().item())

# Sanity: visible + inpaint should never exceed real residues
real = batch["mask"].sum().item()
vis = batch["visible_mask"].sum().item()
inp = batch["inpaint_mask"].sum().item()
print("real vs (visible + inpaint):", real, vis + inp)

# Check torsion range roughly (-pi, pi)
if batch["torsion_angles"].numel() > 0:
    print("torsion min/max:", batch["torsion_angles"].min().item(), batch["torsion_angles"].max().item())
