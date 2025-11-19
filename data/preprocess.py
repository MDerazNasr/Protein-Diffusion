"""Data preprocessing utilities.

	•	PDBParser → reads the text file
	•	loop through model → chain → residue → this is the hierarchy inside a protein
	•	if not residue.has_id("CA") → skip residues missing backbone atoms
	•	get_array() → returns coordinate vector
	•	append them in order
	•	return shape (L residues, 3 atoms, 3 coords)

This is exactly the same representation used in RFdiffusion.
"""

from Bio.PDB import PDBParser #to read .pdb protein structure files
import numpy as np
import os

def load_backbone_coords(pdb_path):
	parser = PDBParser(QUIET=True)
	#Loads the file into a hierarchical object: → models → chains → residues → atoms
	structure = parser.get_structure("protein", pdb_path)
	backbone = []
    
	#iterate over models → chains → residues → atoms
	for model in structure:
		for chain in model:
			for residue in chain:
				#Skip non-standard amino acids, water, ligands
				if not residue.has_id("N") or not residue.has_id("CA") or not residue.has_id("C"):
					continue #skip water molecules, ligands, missing atom residues, skip weird amino acids
							#we only keep residues that have all backbone atoms.
				#extract coordinates
				'''
				[x,y,z] --> each of these is a 3 element numpy array
				residue["N] gets the atom obj and .getvector.getarray() turns it into
				actual coord. array			
				'''
				N = residue["N"].get_vector().get_array()
				CA = residue["CA"].get_vector().get_array()
				C = residue["C"].get_vector().get_array()
				
				#After parsing the whole PDB file you end up with a Python list of shape:
				backbone.append([N, CA, C])
			'''
			[
			[N, CA, C],   ← residue 1
			[N, CA, C],   ← residue 2
			...
			]
			'''
	if len(backbone) == 0:
		raise ValueError(f"No backbone atoms found in {pdb_path}")
	return np.array(backbone) #shape - (L,3,3)

def process_folder(input_folder, output_folder):
	'''
	Process all PDB files in a folder, save as .npy arrays
	'''
	os.makedirs(output_folder, exist_ok=True) #ensure output folder exists
	
	for fname in os.listdir(input_folder):
		if not fname.endswith(".pdb"):
			continue

		try:
			pdb_path = os.path.join(input_folder, fname) #pdb_path is undefined; build the path from input_folder and fname.
			coords = load_backbone_coords(pdb_path) #try to process the file
			#if successful
			out_path = os.path.join(output_folder, fname.replace(".pdb", ".npy"))
			np.save(out_path, coords)
			print(f"[OK] {fname} -> saved to {out_path}")
		#if failed
		except Exception as e:
			print(f"[FAILED] {fname}: {e}")
		
'''
Why is the shape (L,3,3)?

Because diffusion models for protein backbones usually work with:
- L residues
- 3 atoms per residue
- 3D coordinates

Everything is aligned and easier to process

Later could flatten to (L,9) before feeding to the model
'''