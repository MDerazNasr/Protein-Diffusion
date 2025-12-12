'''
Dataset classes for protein data.
PyTorch Dataset + DataLoader for variable length protein length proteins.

Goal:
- Load processes .npy backbone files
- Convert each protein to a Pytorch ready format (Tensor)
- Pad proteins in a batch to the same length
- Create masks so the model knows which positions are valid
- Return mini-batches for training the diffusion models

Note - CA is alpha carbon
'''

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ProteinBackboneDataset(Dataset):
    '''
    Loads individual protein backbones stored as .npy arrays.

    Each array has shape: (L, 3 ,3)
    - L residues
    - 3 backbone atoms (N, CA, C)
    - 3D coordinates
    '''
    def __init__(self, folder_path):
        self.folder_path = folder_path #directory with all processed proteins
        self.files = [] #list of file names ["1CRN.npy", ...]
        for f in os.listdir(folder_path):
            if f.endswith(".npy"):
                self.files.append(f)
        
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {folder_path}")
    
    def __len__(self): #how many files do we have
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.folder_path, fname)
        
        coords = np.load(path) #shape (L, 3, 3)
        
        #Convert to torch Tensor
        coords = torch.tensor(coords, dtype=torch.float32)
        L = coords.shape[0] #protein length

        return {
            "coords": coords, # (L,3,3)
            "length": L, #int
            "name": fname.replace(".npy", "")
        } #for each protein (1 at a time here), return a dict with structure info L,3,3


def protein_collate_fn(batch):
    '''
    Custom collate function to handle variable-length proteins in a batch

    batch is a list of dicts returned by __getitem__

    OUTPUT:
    coords_padded: Tensor shape (B, Lmax, 3, 3)
    ca_coords:      Tensor (B, Lmax, 3)     -> padded CA-only coordinates
    pairwise_dist:  Tensor (B, Lmax, Lmax)  -> CA pairwise distance matrix
    bond_vecs:      Tensor (B, Lmax-1, 3)   -> CA(i)->CA(i+1) backbone vectors
    mask:          Tensor shape (B, Lmax)
    lengths:       Tensor shape (B,) (original lengths)
    names:         list of strings
    '''

    #step 1: Extract Lengths
    lengths = []
    names = []
    for item in batch:
        lengths.append(item["length"])
        names.append(item["name"])
    
    B = len(batch)
    Lmax = max(lengths)

    #Step 2: Create padded tensors
    coords_padded = torch.zeros((B, Lmax, 3, 3), dtype=torch.float32)
    mask = torch.zeros((B, Lmax), dtype=torch.bool)

    #Step 3: Fill in real values
    for i, item in enumerate(batch):
        L = item["length"]
        coords = item["coords"] #shape (l,3,3)

        coords_padded[i, :L, :, :] = coords
        mask[i, :L] = 1 #marks real residues
    '''
    Imagine batch of 2 proteins:
    •	P0: length 5
    •	P1: length 3
    •	Lmax = 5
    we get:
        coords_padded[0, 0:5] = P0 coords
        mask[0] = [1,1,1,1,1]

        coords_padded[1, 0:3] = P1 coords
        coords_padded[1, 3:5] = zeros (kept from initialization)
        mask[1] = [1,1,1,0,0]
    So now we have:
    •	coords_padded.shape == (B, Lmax, 3, 3)
    •	mask.shape == (B, Lmax)

    '''
    lengths = torch.tensor(lengths, dtype=torch.long)
    # After padding coords_padded
    ca_coords = coords_padded[:, :, 1, :]  # (B, L, 3)

    # Step 5: Compute geometry features (mask-aware)
    pairwise_dist = pairwise_distances(ca_coords, mask)      # (B, Lmax, Lmax)
    bond_vecs = backbone_vectors(ca_coords, mask)            # (B, Lmax-1, 3)

    return {
        "coords": coords_padded,
        "ca_coords": ca_coords,
        "pairwise_dist": pairwise_dist,
        "bond_vecs": bond_vecs,
        "mask": mask,
        "lengths": lengths,
        "names": names,
    }


def create_dataloader(folder_path, batch_size=4, shuffle=True, num_workers=0):
    '''
    Main function to create a PyTorch DataLoader for backbones.
    '''
    dataset = ProteinBackboneDataset(folder_path) #tells pytorch how to load single protein
    '''
    •	DataLoader:
    •	randomly picks batch_size indices
    •	calls __getitem__ on each
    •	passes the list of samples to protein_collate_fn
    •	gets back a nice batch dict with padding + masks
    '''
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=protein_collate_fn
    )

    return loader

'''
Why do we need padding?
    PyTorch wants batches to be single tensors.
    If you ask for batch_size = 2, ideally you’d like a tensor:
    coords.shape == (B, L, 3, 3)
    # B = batch size
    # L = same length for all proteins in the batch
    But real proteins are variable-length:
        •	Protein 1: L = 5
        •	Protein 2: L = 8

    You cannot stack them directly; PyTorch will complain:

    “torch.stack: Sizes of tensors must match”

    So we fix this by:
        1.	Finding the longest protein in the batch → Lmax
        2.	Expanding all proteins to that length by adding zeros at the end
    (this is called padding)

    Example with super tiny data:
        •	P1 coords: shape (5, 3, 3)
        •	P2 coords: shape (3, 3, 3)

    We choose Lmax = 5.

    We build a batch tensor (B, Lmax, 3, 3):
        •	For P1:
        •	Fill positions 0..4 with real values
        •	For P2:
        •	Fill positions 0..2 with real values
        •	Fill positions 3..4 with zeros (fake / padding)

    Now PyTorch can handle this as a single tensor.

    ⸻

    Why do we need a mask?

    Padding creates fake residues (zeros you added to make sizes match).

    The model must not:
        •	compute loss on them
        •	treat them as real residues
        •	use them when computing things like RMSD, distances, etc.

    So we create a mask that says:
        •	True (or 1) → this position is a real residue
        •	False (or 0) → this is padding, ignore

    For the tiny example:
        •	P1 (length 5): [1, 1, 1, 1, 1]
        •	P2 (length 3): [1, 1, 1, 0, 0]

    Batch mask: shape (B, Lmax).

    Your model can do stuff like: 
    loss = (per_residue_loss * mask).sum() / mask.sum()
'''