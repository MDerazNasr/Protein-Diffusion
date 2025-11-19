"""Sequence conditioning module for diffusion models."""

class SequenceConditioner:
    def __init__(self, config=None):
        """
        TODO:
        - Extract backbone features (distances, angles, dihedrals)
        - Build transformer or equivariant-conditioned decoder
        - Train on (structure -> sequence) recovery task
        - Add sampling for sequence design on generated backbones
        """
        pass

    def forward(self, backbone_feats):
        """
        TODO:
        - Predict amino acid logits at each residue
        - Return distribution for teacher forcing
        """
        raise NotImplementedError