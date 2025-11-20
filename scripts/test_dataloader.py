import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import create_dataloader

data_dir = PROJECT_ROOT / "data" / "processed"
loader = create_dataloader(str(data_dir), batch_size=2)

for batch in loader:
    print("Coords shape:", batch["coords"].shape)
    print("Mask shape:", batch["mask"].shape)
    print("Lengths:", batch["lengths"])
    print("Names:", batch["names"])
    break