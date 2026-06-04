#!/usr/bin/env python3
"""
scMaize: Single-cell Maize Foundation Model
============================================
Model architecture V2 with:
- LayerNorm after embeddings
- CLS token for better embeddings
- Batch-aware training (optional)
- Contrastive learning (optional)

Two variants:
1. scMaizeExp: Expression-only model
2. scMaizeGO: Expression + GO embedding model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=mask)


class scMaizeBase(nn.Module):
    """Base class for scMaize models"""
    
    def __init__(
        self,
        vocab_size: int = 15000,
        d_model: int = 384,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
        use_func_embedding: bool = False,
        use_batch_label: bool = True,
        use_contrastive: bool = False,  # Default disabled to preserve biological variation
        temperature: float = 0.1,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_func_embedding = use_func_embedding
        self.use_batch_label = use_batch_label
        self.use_contrastive = use_contrastive
        self.temperature = temperature
        
        # Gene Embedding + LayerNorm
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.gene_norm = nn.LayerNorm(d_model)
        
        # Expression Projection + LayerNorm (FIXED)
        self.expr_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # GO Embedding Projection
        if use_func_embedding:
            self.func_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            )
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=8000, dropout=dropout)
        
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # CLS Token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Batch-aware embedding
        if use_batch_label:
            self.batch_embedding = nn.Embedding(256, d_model)
            self.batch_norm = nn.LayerNorm(d_model)
        
        # Output LayerNorm
        self.output_norm = nn.LayerNorm(d_model)
        
        # MGM Head
        self.mlm_head = nn.Linear(d_model, 1)
        
        # Projection head for contrastive learning
        if use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.cls_token, std=0.02)
    
    def set_func_embedding(self, embedding: torch.Tensor):
        self.register_buffer('func_embed', embedding)
        self.func_embed.requires_grad = False
    
    def _get_cell_embedding(
        self,
        hidden: torch.Tensor,
        batch_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        cell_emb = hidden[:, 0, :]
        cell_emb = self.output_norm(cell_emb)
        
        return cell_emb
    
    def forward(
        self,
        gene_ids: torch.Tensor,
        expr_values: torch.Tensor,
        batch_labels: Optional[torch.Tensor] = None,
        return_emb: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = gene_ids.size(0)
        
        # Gene embedding + LayerNorm
        gene_embed = self.gene_norm(self.gene_embedding(gene_ids))
        
        # Expression embedding + LayerNorm
        expr_embed = self.expr_projection(expr_values.unsqueeze(-1))
        
        x = gene_embed + expr_embed
        
        # Add batch embedding BEFORE transformer (affects all positions)
        if self.use_batch_label and batch_labels is not None:
            batch_emb = self.batch_norm(self.batch_embedding(batch_labels))
            x = x + batch_emb.unsqueeze(1)
        
        # Add GO embedding
        if self.use_func_embedding and hasattr(self, 'func_embed'):
            func_embed = F.embedding(gene_ids, self.func_embed)
            func_embed = self.func_proj(func_embed)
            x = x + func_embed
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.pos_encoder(x)
        
        hidden = self.transformer(x)
        
        # MGM prediction (skip CLS position)
        logits = self.mlm_head(hidden[:, 1:, :]).squeeze(-1)
        
        cell_emb = None
        if return_emb:


            cell_emb = self._get_cell_embedding(hidden, batch_labels)
        
        return logits, cell_emb
    
    def get_projection(self, cell_emb: torch.Tensor) -> torch.Tensor:
        if self.use_contrastive:
            return self.projection_head(cell_emb)
        return cell_emb


class scMaizeExp(scMaizeBase):
    """scMaize Expression-only model"""
    
    def __init__(
        self,
        vocab_size: int = 15000,
        d_model: int = 384,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
        use_batch_label: bool = True,
        use_contrastive: bool = False,  # Default disabled to preserve biological variation
        temperature: float = 0.1,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_func_embedding=False,
            use_batch_label=use_batch_label,
            use_contrastive=use_contrastive,
            temperature=temperature,
            use_gradient_checkpointing=use_gradient_checkpointing
        )


class scMaizeGO(scMaizeBase):
    """scMaize Expression + GO embedding model"""
    
    def __init__(
        self,
        vocab_size: int = 15000,
        d_model: int = 384,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
        use_batch_label: bool = True,
        use_contrastive: bool = False,  # Default disabled to preserve biological variation
        temperature: float = 0.1,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_func_embedding=True,
            use_batch_label=use_batch_label,
            use_contrastive=use_contrastive,
            temperature=temperature,
            use_gradient_checkpointing=use_gradient_checkpointing
        )


class MGMLoss(nn.Module):
    """Masked Gene Expression Modeling Loss with optional non-zero weighting
    
    Features:
    - Weighted MSE for non-zero values (to address sparse data)
    - Prevents model from predicting all zeros
    - Retains biological dynamic range
    """
    def __init__(self, nonzero_weight: float = 5.0, use_weighted: bool = True):
        super().__init__()
        self.nonzero_weight = nonzero_weight
        self.use_weighted = use_weighted
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        # Base MSE loss
        loss = (pred - target) ** 2
        
        if self.use_weighted:
            # Non-zero value weighting to prevent "all-zero" prediction
            # In single-cell data, >90% values are zeros
            # Giving higher weight to non-zero values helps retain biological signal
            nonzero_mask = (target > 0).float()
            weight = 1.0 + (self.nonzero_weight - 1.0) * nonzero_mask
            loss = loss * weight
        
        # Apply mask and compute mean
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for batch-aware training"""
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        batch_labels: torch.Tensor
    ) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        batch_size = embeddings.size(0)
        labels = batch_labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        mask_pos = mask.sum(dim=1)
        mask_pos = torch.where(mask_pos > 0, mask_pos, torch.ones_like(mask_pos))
        
        loss = -(mask * log_prob).sum(dim=1) / mask_pos
        return loss.mean()


def create_model(
    model_type: str = 'scMaizeExp',
    vocab_size: int = 15000,
    d_model: int = 384,
    n_heads: int = 4,
    n_layers: int = 6,
    d_ff: int = 1536,
    dropout: float = 0.1,
    use_batch_label: bool = True,
    use_contrastive: bool = False,  # Default disabled to preserve biological variation
    temperature: float = 0.1,
    func_embed_path: Optional[str] = None,
    use_gradient_checkpointing: bool = False,
    device: str = 'cuda'
) -> scMaizeBase:
    """Model factory"""
    import pickle
    
    if model_type == 'scMaizeExp':
        model = scMaizeExp(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_batch_label=use_batch_label,
            use_contrastive=use_contrastive,
            temperature=temperature,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
    elif model_type == 'scMaizeGO':
        model = scMaizeGO(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_batch_label=use_batch_label,
            use_contrastive=use_contrastive,
            temperature=temperature,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if model_type == 'scMaizeGO' and func_embed_path is not None:
        with open(func_embed_path, 'rb') as f:
            data = pickle.load(f)
        func_embed = torch.tensor(data['embedding'], dtype=torch.float32)
        model.set_func_embedding(func_embed)
        print(f"Loaded GO embedding: {func_embed.shape}")
    
    model = model.to(device)
    return model


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test scMaizeExp
    model = create_model('scMaizeExp', vocab_size=15000, d_model=384, n_heads=4, n_layers=6, device=device)
    
    batch_size = 4
    seq_len = 2048
    gene_ids = torch.randint(0, 15000, (batch_size, seq_len)).to(device)
    expr_values = torch.randn(batch_size, seq_len).abs().to(device)
    batch_labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    logits, cell_emb = model(gene_ids, expr_values, batch_labels=batch_labels)
    
    print(f"scMaizeExp test:")
    print(f"  Output shape: {logits.shape}")
    print(f"  Cell embedding shape: {cell_emb.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
