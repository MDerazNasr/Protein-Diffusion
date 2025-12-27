import torch
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data.dataset import create_dataloader
from eval.metrics import compute_backbone_metrics

loader = create_dataloader("data/processed", batch_size=1, shuffle=False)
batch = next(iter(loader))

ca = batch["ca_coords"]
mask = batch["mask"]

metrics = compute_backbone_metrics(ca, mask)
print(metrics)
