import torch

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import create_dataloader
from models.diffusion_backbone import BackboneDiffusionModel

device = "cuda" if torch.cuda.is_available() else "cpu"

loader = create_dataloader("data/processed", batch_size=2, shuffle=False)
batch = next(iter(loader))

x0 = batch["ca_coords"].to(device)
mask = batch["mask"].to(device)
inpaint_mask = batch.get("inpaint_mask", None)
if inpaint_mask is not None:
    inpaint_mask = inpaint_mask.to(device)

model = BackboneDiffusionModel(T=100).to(device)
model.train()

loss, base_loss, bond_loss, clash_loss = model.training_loss(x0, mask, inpaint_mask=inpaint_mask)
print("Day 6 sanity loss:", float(loss.item()))
print(f"  Base: {base_loss.item():.3f}, Bond: {bond_loss.item():.3f}, Clash: {clash_loss.item():.3f}")
