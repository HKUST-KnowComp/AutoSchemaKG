# Installation Guide

This guide provides detailed instructions for installing the `atlas-rag` package and its dependencies.

## Requirements

- **Python**: 3.9 or higher
- **Operating Systems**: Linux, macOS, Windows

## Prerequisites

Before installing `atlas-rag`, you must install PyTorch and FAISS manually to ensure hardware compatibility.

### 1. Install PyTorch

Please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the installation command appropriate for your system.

Example for Linux:
```bash
# For CPU-only systems
pip install torch torchvision torchaudio

# For systems with NVIDIA GPUs (adjust CUDA version as needed)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

### 2. Install FAISS

You need to install either the CPU or GPU version of [FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md):
(Recommend to install faiss with CUDA-12.6)
```bash
# For CPU-only systems
pip install faiss-cpu

# For systems with NVIDIA GPUs (adjust CUDA version as needed)
conda install -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version=12.6'
```

## Basic Installation

### Install from PyPI

The simplest way to install `atlas-rag` is via pip:

```bash
pip install atlas-rag
```

This will install the core package with all required dependencies.

## Optional Dependencies

### NV-Embed-v2 Support

If you need support for NVIDIA's NV-embed-v2 model, install the package with the `nvembed` extra:

```bash
pip install atlas-rag[nvembed]
```

This installs compatible versions of `transformers` (>=4.42.4, <=4.47.1) and `sentence-transformers` (2.7.0) required for NV-embed-v2.

## Verification

After installation, verify that the package is installed correctly:

```python
import atlas_rag
print(atlas_rag.__version__)
```

You can also verify the installation of key components:

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.kg_construction.triple_config import ProcessingConfig
print("Installation successful!")
```

## Development Installation

If you want to contribute to the project or install from source:

1. Clone the repository:
```bash
git clone https://github.com/HKUST-KnowComp/AutoSchemaKG.git
git checkout release/v0.0.5 # checkout to your desired branch
cd AutoSchemaKG
```

2. Install in development mode:
```bash
pip install -e .
```

## Next Steps

After successful installation:

1. Check out the [Quick Start Guide](quickstart.md) to begin using atlas-rag
2. Explore [Examples](../example/advance_features.md) for advance use cases

## Support

If you encounter any installation issues:

- Check the [GitHub Issues](https://github.com/HKUST-KnowComp/AutoSchemaKG/issues)
- Contact the maintainers:
  - Dennis Hong Ting TSANG: httsangaj@connect.ust.hk
  - Jiaxin Bai: jbai@connect.ust.hk   
  - Haoyu Huang: haoyuhuang@link.cuhk.edu.hk
