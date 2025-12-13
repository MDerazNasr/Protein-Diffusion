import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import create_dataloader

loader = create_dataloader(PROJECT_ROOT / "data" / "processed", batch_size=2)

batch = next(iter(loader))

print("CA coords:", batch["ca_coords"].shape)
print("Pairwise dist:", batch["pairwise_dist"].shape)
print("Bond vectors:", batch["bond_vecs"].shape)
