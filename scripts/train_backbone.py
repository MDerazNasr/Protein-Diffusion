import os
import torch
from torch.optim import AdamW
from tqdm import tqdm

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import create_dataloader
from models.diffusion_backbone import BackboneDiffusionModel


def main():
    # Config (keep it simple for now)
    processed_folder = "data/processed"
    batch_size = 4
    lr = 5e-5
    epochs = 5
    T = 500  # diffusion steps (smaller for faster dev)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("checkpoints", exist_ok=True)

    # Data
    loader = create_dataloader(processed_folder, batch_size=batch_size, shuffle=True)
    # Model
    model = BackboneDiffusionModel(T=T).to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim = AdamW(model.parameters(), lr=lr)
    # Train
    model.train()
    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            # Batch contains: ca_coords, mask, inpaint_mask, etc. from Day 5 collate
            x0 = batch["ca_coords"].to(device)          # (B, L, 3)
            mask = batch["mask"].to(device)             # (B, L)
            inpaint_mask = batch.get("inpaint_mask", None)
            if inpaint_mask is not None:
                inpaint_mask = inpaint_mask.to(device)

            loss, base, bond, clash = model.training_loss(
                x0,
                mask,
                inpaint_mask=inpaint_mask
            )

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()

            pbar.set_postfix(
                loss=float(loss.item()),
                base=float(base.item()),
                bond=float(bond.item()),
                clash=float(clash.item()),
            )


            global_step += 1
            pbar.set_postfix(loss=float(loss.item()))

        # checkpoint each epoch
        ckpt_path = f"checkpoints/backbone_diffusion_epoch{epoch+1}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "epoch": epoch + 1,
                "T": T,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

    print("Training done.")


if __name__ == "__main__":
    main()
