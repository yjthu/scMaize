#!/usr/bin/env python3
"""
scMaize Embedding Extraction Script
===================================
Generate cell embeddings from trained scMaize models.

Usage:
    # Extract embeddings using scMaizeExp model
    python generate_embeddings.py --exp scMaizeExp --input.h5ad --output embeddings.npy
    
    # Extract embeddings using scMaizeGO model
    python generate_embeddings.py --exp scMaizeGO --input.h5ad --output embeddings.npy
    
    # Use batch correction (remove batch embedding)
    python generate_embeddings.py --exp scMaizeExp --input.h5ad --output embeddings.npy --remove-batch
"""

import argparse
import os
import sys
import random
import torch
import numpy as np
import anndata as ad
import gc
from scipy import sparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from model_scmaize import create_model

SEED = 42


def load_checkpoint(model_path: str, model_type: str, device: str):
    """Load model from checkpoint"""
    print(f"Loading model: {model_type}")
    
    # Create model
    model = create_model(
        model_type=model_type,
        vocab_size=15000,
        d_model=384,
        n_heads=4,
        n_layers=6,
        d_ff=1536,
        dropout=0.1,
        use_batch_label=True,
        use_contrastive=False,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model


def prepare_h5ad_data(h5ad_path: str, gene_list: list, batch_size: int = 512):
    """Load and prepare h5ad data for embedding extraction"""
    print(f"Loading h5ad: {h5ad_path}")
    
    # Load h5ad
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}")
    
    # Build gene name to index mapping
    h5ad_genes = adata.var.index.tolist()
    gene_to_idx = {g: i for i, g in enumerate(h5ad_genes)}
    
    # Find target gene indices
    gene_indices = []
    missing_genes = []
    for gene in gene_list:
        if gene in gene_to_idx:
            gene_indices.append(gene_to_idx[gene])
        else:
            missing_genes.append(gene)
    
    if missing_genes:
        print(f"  Warning: {len(missing_genes)} genes not found, using {len(gene_indices)}/{len(gene_list)}")
    
    gene_indices = np.array(gene_indices, dtype=np.int64)
    
    # Auto-detect data type and apply appropriate transformation
    X = adata.X
    
    # Try to use count layer if available (common in anndata)
    use_count_layer = False
    if 'count' in adata.layers or 'counts' in adata.layers:
        count_key = 'count' if 'count' in adata.layers else 'counts'
        X = adata.layers[count_key]
        use_count_layer = True
        print(f"  Using '{count_key}' layer for expression values")
    
    if sparse.issparse(X):
        X_dense = X[:, gene_indices].toarray()
    else:
        X_dense = X[:, gene_indices]
    
    # Auto-detect if log transformation is needed
    # Heuristics:
    # - Raw counts: typically integers, larger range (max >> 10)
    # - Log-transformed: typically floats, smaller range (max < 15)
    max_val = X_dense.max()
    is_integer = np.issubdtype(X_dense.dtype, np.integer)
    
    need_log = True
    if is_integer and max_val > 20:
        # Likely raw counts (large integer values)
        need_log = True
        print(f"  Detected raw counts: max={max_val:.0f}, applying log1p")
    elif not is_integer and max_val < 15:
        # Likely already log-transformed (small float values)
        need_log = False
        print(f"  Detected log-transformed data: max={max_val:.2f}, using as-is")
    else:
        # Ambiguous case
        if max_val < 15:
            need_log = False
            print(f"  Detected log-transformed (ambiguous): max={max_val:.2f}, using as-is")
        else:
            need_log = True
            print(f"  Detected raw counts (ambiguous): max={max_val:.0f}, applying log1p")
    
    if need_log:
        X_log = np.log1p(X_dense).astype(np.float32)
    else:
        X_log = X_dense.astype(np.float32)
    
    # Get batch labels if available
    if 'batch' in adata.obs.columns:
        batch_labels = adata.obs['batch'].values
        batch_to_int = {b: i for i, b in enumerate(np.unique(batch_labels))}
        batch_labels_int = np.array([batch_to_int[b] for b in batch_labels])
    else:
        batch_labels_int = np.zeros(len(adata), dtype=np.int64)
    
    n_cells = X_log.shape[0]
    print(f"  Prepared: {n_cells} cells, {len(gene_indices)} genes")
    
    del adata, X, X_dense
    gc.collect()
    
    return X_log, batch_labels_int, gene_indices


def extract_embeddings(
    model,
    expr_data: np.ndarray,
    batch_labels: np.ndarray,
    gene_indices: np.ndarray,
    gene_list: list,
    device: str,
    batch_size: int = 512,
    seq_len: int = 2048,
    remove_batch: bool = False
):
    """Extract cell embeddings from model"""
    print(f"\nExtracting embeddings (batch_size={batch_size})...")
    
    n_cells = expr_data.shape[0]
    n_genes = len(gene_list)
    
    # Pre-generate random gene indices for each cell (consistent with training)
    # Use fixed seed for reproducibility
    rng = np.random.RandomState(42)
    all_gene_indices = np.array([
        sorted(rng.choice(n_genes, seq_len, replace=False))
        for _ in range(n_cells)
    ], dtype=np.int64)
    
    all_embeddings = []
    
    for i in range(0, n_cells, batch_size):
        end_idx = min(i + batch_size, n_cells)
        batch_expr = expr_data[i:end_idx]
        batch_gene_idx = all_gene_indices[i:end_idx]
        batch_batch = batch_labels[i:end_idx]
        
        # Get gene IDs
        gene_ids = batch_gene_idx
        expr_values = np.array([
            batch_expr[j, batch_gene_idx[j]]
            for j in range(len(batch_expr))
        ])
        
        # Convert to tensors
        gene_ids_tensor = torch.tensor(gene_ids, dtype=torch.long).to(device)
        expr_tensor = torch.tensor(expr_values, dtype=torch.float32).to(device)
        
        if remove_batch:
            batch_labels_tensor = None
        else:
            batch_labels_tensor = torch.tensor(batch_batch, dtype=torch.long).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            _, cell_emb = model(
                gene_ids_tensor,
                expr_tensor,
                batch_labels=batch_labels_tensor,
                return_emb=True
            )
            embeddings = cell_emb.cpu().numpy()
        
        all_embeddings.append(embeddings)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Progress: {end_idx}/{n_cells}")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description='scMaize Embedding Extraction')
    parser.add_argument('--exp', type=str, required=True,
                        choices=['scMaizeExp', 'scMaizeGO'],
                        help='Experiment type')
    parser.add_argument('--input', type=str, required=True,
                        help='Input h5ad file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output embeddings file path (.npy)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path (default: auto from checkpoints/)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for embedding extraction')
    parser.add_argument('--remove-batch', action='store_true',
                        help='Remove batch embedding from final embeddings')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--flash-sdp', action='store_true',
                        help='启用 PyTorch SDPA 加速 (需要 PyTorch 2.0+)')
    
    args = parser.parse_args()
    # 固定随机种子，确保可复现性
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    # 启用 PyTorch SDPA 加速
    if args.flash_sdp and torch.cuda.is_available():
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("已启用 PyTorch SDPA 加速")
        else:
            print("警告: PyTorch 版本不支持 SDPA，请升级到 2.0+")
    else:
        print("使用标准 Attention")
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load gene list
    data_path = os.path.join(PROJECT_DIR, 'data', 'preprocessed_data.pkl')
    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    gene_list = data['gene_list']
    print(f"Gene list: {len(gene_list)} genes")
    
    # Determine checkpoint path
    if args.checkpoint is None:
        checkpoint_dir = os.path.join(PROJECT_DIR, 'checkpoints')
        prefix = 'scMaizeExp' if args.exp == 'scMaizeExp' else 'scMaizeGO'
        
        # Find all checkpoints with this prefix
        import glob
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, f'{prefix}_epoch*.pt')))
        
        if checkpoints:
            # Use the latest epoch checkpoint
            checkpoint_path = checkpoints[-1]
            print(f"Using checkpoint: {os.path.basename(checkpoint_path)}")
        else:
            # Fallback to old naming
            checkpoint_path = os.path.join(checkpoint_dir, f'{prefix}_best.pt')
    else:
        checkpoint_path = args.checkpoint
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model = load_checkpoint(checkpoint_path, args.exp, device)
    
    # Prepare h5ad data
    expr_data, batch_labels, gene_indices = prepare_h5ad_data(
        args.input, gene_list, args.batch_size
    )
    
    # Extract embeddings
    embeddings = extract_embeddings(
        model,
        expr_data,
        batch_labels,
        gene_indices,
        gene_list,
        device,
        batch_size=args.batch_size,
        remove_batch=args.remove_batch
    )
    
    # Save embeddings
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.save(args.output, embeddings)
    print(f"\nSaved embeddings to: {args.output}")
    print(f"Shape: {embeddings.shape}")
    print(f"dtype: {embeddings.dtype}")
    
    if args.remove_batch:
        print("\nNote: Batch embeddings have been removed from the output.")


if __name__ == "__main__":
    main()
