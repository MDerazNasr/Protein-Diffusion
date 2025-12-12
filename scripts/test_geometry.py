from data.dataset import create_dataloader

loader = create_dataloader("data/processed", batch_size=2)

batch = next(iter(loader))

print("CA coords:", batch["ca_coords"].shape)
print("Pairwise dist:", batch["pairwise_dist"].shape)
print("Bond vectors:", batch["bond_vecs"].shape)
