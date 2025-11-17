# Protein Diffusion

A deep learning framework for protein structure generation using diffusion models.


## 🔭 Project Vision

ProteinDiffusion aims to build a modern, research-grade generative protein design system inspired by RFdiffusion and ProteinMPNN.  
This project combines **SE(3)-equivariant diffusion models**, **conditional sequence generation**, **CUDA/C++ geometric kernels**, and **AlphaFold/OpenFold-based validation** into a fully integrated pipeline.

The final system will support:

- **Unconditional backbone generation** using an SE(3)-equivariant diffusion process  
- **Structure inpainting**, allowing the model to fill in masked or missing backbone regions  
- **Conditional sequence design**, generating amino acid sequences tailored to generated backbones  
- **CUDA-accelerated structural evaluation**, including distance matrices, RMSD, steric clash counts, and geometric validity  
- **AlphaFold/OpenFold plausibility scoring** for biologically relevant validation  
- **Interactive Protein Quality Dashboard** for 3D visualization and structural diagnostics  
- **End-to-end protein generation demos**, including latent interpolation and backbone morphing  

This repository will evolve into a clean, modular codebase suitable for ML-for-biology research, GPU kernel optimization, and academic demonstrations in computational structural biology.
## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Usage

See `scripts/` for training and generation scripts.

## License

MIT

