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
        coords = torch.Tensor(coords, dtype=torch.float32)
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

