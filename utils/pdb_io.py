#Export CA coordinates to a PDB file
'''
A PDB file is text.
Each line describes an atom.
write 1 CA atom per residue.
Residue name is set to ALA as a placeholder.
'''

import numpy as np

def write_ca_pdb(ca_coords, out_path, chain_id="A"):
    '''
    Write CA-only coordinates to a minimal PDB.

    Args:
        ca_coords: (L,3) numpy array or torch tensor
        out_path: output filename
    '''

    if hasattr(ca_coords, "detach"):
        ca_coords = ca_coords.detach().cpu().numpy()

    ca_coords = np.asarray(ca_coords)
    L = ca_coords.shape[0]

    lines = []
    atom_serial = 1

    for i in range(L):
        x, y, z = ca_coords[i]
        res_seq = i + 1

        #CA-only residue; use ALA as placeholder amino acid
        #PDB ATOM line formatting: strict columns but this works in most viewers
        line = (
            f"ATOM  {atom_serial:5d}  CA  ALA {chain_id}{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        lines.append(line)
        atom_serial += 1

    lines.append("END")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))        