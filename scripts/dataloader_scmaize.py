#!/usr/bin/env python3
"""
scMaize DataLoader
==================
Data loader for scMaize training with batch labels support
Features:
- Each epoch re-randomizes gene selection per cell (sorted, consistent with V1)
- Each epoch re-randomizes mask positions
- Batch labels support for batch-aware training
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, Optional


class SingleCellDataset(Dataset):
    """Single cell dataset for scMaize training"""
    
    def __init__(
        self,
        expr_data: np.ndarray,
        gene_list: list,
        seq_len: int = 2048,
        mask_ratio: float = 0.15,
        batch_labels: Optional[np.ndarray] = None,
        n_genes_total: int = 15000,
        seed: int = 42
    ):
        self.expr_data = expr_data
        self.gene_list = gene_list
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.n_genes_total = n_genes_total
        self.seed = seed
        self.batch_labels = batch_labels
        
        self.n_samples = len(expr_data)
        self.n_mask = int(seq_len * mask_ratio)
        
        # Base seed for reproducibility (increment each regenerate)
        self.base_seed = seed
        
        # Gene selection and mask storage
        self.selected_gene_indices = None
        self.mask_indices = None
        
        # Don't pre-generate in __init__ - lazy generation when needed
        # self.regenerate_gene_indices()
        # self.regenerate_mask()
        
        self._current_seed = seed
    
    def regenerate_gene_indices(self):
        """每个 epoch 为每个细胞重新随机选择基因 (排序后使用，与 V1 一致)"""
        rng = np.random.RandomState(self.base_seed)
        
        self.selected_gene_indices = np.array([
            sorted(rng.choice(
                self.n_genes_total,
                self.seq_len,
                replace=False
            ))
            for _ in range(self.n_samples)
        ])
        
        # Update seed for next regenerate
        self.base_seed += 1
    
    def regenerate_mask(self):
        """每个 epoch 重新生成固定 mask 位置 (与 V1 一致)"""
        rng = np.random.RandomState(self.base_seed + 1000)
        self.mask_indices = sorted(rng.choice(self.seq_len, self.n_mask, replace=False))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Lazy generation if not pre-generated
        if self.selected_gene_indices is None:
            self.regenerate_gene_indices()
        if self.mask_indices is None:
            self.regenerate_mask()
        
        gene_indices = self.selected_gene_indices[idx]
        expr_values = self.expr_data[idx, gene_indices]
        
        # Convert to tensors
        gene_ids = torch.tensor(gene_indices, dtype=torch.long)
        expr_values_tensor = torch.tensor(expr_values, dtype=torch.float32)
        
        # MGM target: clone expression values
        mgm_target = expr_values_tensor.clone()
        
        # MGM mask: create mask and zero out masked positions
        mgm_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        mgm_input = expr_values_tensor.clone()
        
        for i in self.mask_indices:
            mgm_mask[i] = True
            mgm_input[i] = 0  # mask position set to 0
        
        # Batch label
        batch_label = 0
        if self.batch_labels is not None:
            batch_label = self.batch_labels[idx]
        
        return {
            'gene_ids': gene_ids,
            'expr_values': mgm_input,
            'mgm_target': mgm_target,
            'mgm_mask': mgm_mask,
            'batch_labels': batch_label,
        }
    
    def set_seed(self, seed: int):
        """从 checkpoint 恢复 seed"""
        self.base_seed = seed
        self._current_seed = seed
    
    def get_seed(self) -> int:
        """获取当前 base_seed，用于 checkpoint 保存"""
        return self.base_seed


def collate_fn(batch):
    """批次整理函数"""
    gene_ids = torch.stack([b['gene_ids'] for b in batch])
    expr_values = torch.stack([b['expr_values'] for b in batch])
    mgm_target = torch.stack([b['mgm_target'] for b in batch])
    mgm_mask = torch.stack([b['mgm_mask'] for b in batch])
    batch_labels = torch.tensor([b['batch_labels'] for b in batch])
    
    return {
        'gene_ids': gene_ids,
        'expr_values': expr_values,
        'mgm_target': mgm_target,
        'mgm_mask': mgm_mask,
        'batch_labels': batch_labels,
    }
