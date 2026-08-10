# 🌽 scMaize

**Single-Cell Foundation Models for Maize**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

scMaize is a family of Transformer-based foundation models pretrained on a comprehensive maize single-cell transcriptomic atlas (385,675 cells × 15,000 genes). Two model variants are provided:

| Model | Description | Parameters |
|-------|-------------|------------|
| **scMaizeExp** | Expression-only foundation model | 16.7M |
| **scMaizeGO** | Expression + Gene Ontology functional embedding | 16.8M |

📄 **Preprint**: [bioRxiv link](https://doi.org/10.64898/2026.08.01.742180)   
🌐 **Project Portal**:  [www.scmaize.com](https://www.scmaize.com)    

---

## 📦 Installation

```bash
git clone https://github.com/your-org/scMaize.git
cd scMaize
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Inference: Generate Cell Embeddings

For inference, you only need the model checkpoint and your own expression data in `.h5ad` format. The training data files are **not required**.

```bash
# Extract embeddings using scMaizeExp
python scripts/generate_embeddings.py \
    --exp scMaizeExp \
    --input /path/to/your/data.h5ad \
    --output embeddings_exp.npy

# Extract embeddings using scMaizeGO
python scripts/generate_embeddings.py \
    --exp scMaizeGO \
    --input /path/to/your/data.h5ad \
    --output embeddings_go.npy

# Remove batch effects (optional)
python scripts/generate_embeddings.py \
    --exp scMaizeGO \
    --input /path/to/your/data.h5ad \
    --output embeddings_go_nobatch.npy \
    --remove-batch
```

**Input requirements**: Your `.h5ad` file should contain:
- `adata.var`: Gene names (must include the 15,000 training genes)
- `adata.X`: Expression matrix (supports sparse format)
- `adata.obs['batch']`: Batch labels (optional, for batch-aware embeddings)

**Output format**: NumPy array of shape `(n_cells, 384)`.

### Training: Train from Scratch

To train scMaize on your own data, you need to prepare a preprocessed data file (see `scripts/preprocess_scmaize.py`) and update `configs/training_config.py` with your data paths and experiment settings.

```bash
# Train scMaizeExp (80 epochs)
python scripts/run_training.py --exp scMaizeExp --epochs 80

# Train scMaizeGO
python scripts/run_training.py --exp scMaizeGO --epochs 80

# Resume from checkpoint
python scripts/run_training.py --exp scMaizeExp \
    --resume checkpoints/scMaizeExp_epoch40.pt \
    --epochs 100
```

For the original paper results, the pretraining data (~22 GB) and pretrained checkpoints are available for download:

| File | Size | Description |
|------|------|-------------|
| `preprocessed_data.pkl` | 21.6 GB | Preprocessed expression matrix + metadata |
| `M3_improved_gene_embedding.pkl` | ~30 MB | GO functional embeddings (M3 BERT method) |
| `scMaizeExp_best.pt` | 75.3 MB | scMaizeExp checkpoint (epoch 55) |
| `scMaizeGO_best.pt` | 97.8 MB | scMaizeGO checkpoint (epoch 65) |

---

## 🏗️ Model Architecture

scMaize is a 6-layer Transformer encoder with four specialized input branches:

```
Input Sequence (2,048 genes, sorted by HVG rank + CLS token)
    │
    ├── Gene Identity Embedding → LayerNorm
    ├── Expression Value → 2-layer MLP (1→384→384) → GELU → LayerNorm
    ├── Batch Label → Embedding (additive bias)
    └── [scMaizeGO only] GO Embedding → Linear Projection → Add
    │
    ▼
┌─────────────────────────────────────────┐
│  Transformer Encoder × 6                │
│  d=384, h=4, FFN=1536, dropout=0.1      │
└─────────────────────────────────────────┘
    │
    ▼
CLS Token → LayerNorm → Cell Embedding (384-dim)
```

| Hyperparameter | Value |
|---------------|-------|
| Vocabulary size | 15,000 genes (13K HVG + 2K GO-supplemented) |
| Sequence length | 2,048 |
| Hidden dimension | 384 |
| Attention heads | 4 |
| Layers | 6 |
| Feed-forward dim | 1,536 |
| Parameters (Exp) | 16.7M |
| Parameters (GO) | 16.8M |

### Pretraining

- **Objective**: Masked Gene Modeling (MGM) with 15% masking
- **Loss**: Weighted MSE (5× on non-zero expression values)
- **Data**: 385,675 cells (80/10/10 train/val/test split)
- **Training**: AdamW, lr=2e-4, batch 256 (eff.), 80 epochs
- **Hardware**: Single NVIDIA A100 40GB (peak VRAM <15 GB)

---

## 📊 Pretrained Model Performance

| Metric | scMaizeExp | scMaizeGO |
|--------|-----------|-----------|
| Global Pearson (MGM) | 0.7825 | 0.7841 |
| Global Spearman (MGM) | 0.4844 | 0.5008 |
| Per-gene PCC (median) | 0.5514 | 0.5541 |
| Cell-type classification acc. | 84.3% | 86.0% |
| Tissue classification acc. | 96.1% | 97.1% |

GO functional embedding provides consistent improvements in rank-order prediction and cell embedding quality, without increasing model size substantially (16.7M → 16.8M parameters).

---

## 🔬 Key Innovations

1. **Species-specific pretraining**: Trained exclusively on maize data, capturing organism-specific regulatory patterns.

2. **GO functional priors**: Gene Ontology embeddings serve as an inductive bias, guiding attention toward functionally coherent gene modules.

3. **Weighted MGM loss**: 5× penalty on non-zero expression values addresses the sparsity challenge of single-cell data.

4. **Additive batch conditioning**: Batch labels are injected as learnable biases that can be removed post-hoc for batch-corrected embeddings.

5. **Parameter-efficient design**: 16.7–16.8M parameters, trainable on a single consumer GPU.

---

## 📝 Citation

If you use scMaize in your research, please cite:

```
Cheng Q.*, Zhang Y.*, Wu T., Zhao A., Shang M., Wang X., Yan J.
scMaize: A Single-Cell Foundation Model and Integrated Atlas for Maize.
bioRxiv, 2026. DOI:10.64898/2026.08.01.742180
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
