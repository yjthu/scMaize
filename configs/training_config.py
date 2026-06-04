#!/usr/bin/env python3
"""
scMaize Training Configuration
==============================
Configuration for scMaizeExp and scMaizeGO training
"""

import os

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = PROJECT_DIR

PATH_CONFIG = {
    'project_dir': BASE_DIR,
    'data_dir': os.path.join(BASE_DIR, 'data'),
    'preprocessed_data': os.path.join(BASE_DIR, 'data', 'preprocessed_data.pkl'),
    'log_dir': os.path.join(BASE_DIR, 'logs'),
    'checkpoint_dir': os.path.join(BASE_DIR, 'checkpoints'),
    'experiment_dir': os.path.join(BASE_DIR, 'experiments'),
    'figure_dir': os.path.join(BASE_DIR, 'figures'),
    'go_embedding_path': os.path.join(
        BASE_DIR, 'data', 'M3_improved_gene_embedding.pkl'
    )
}

# Model configuration
MODEL_CONFIG = {
    'vocab_size': 15000,
    'hidden_dim': 384,
    'num_heads': 4,
    'num_layers': 6,
    'mlp_expansion': 4,
    'dropout': 0.1,
}

# Data configuration
DATA_CONFIG = {
    'seq_len': 2048,
    'mask_ratio': 0.15,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 2e-4,
    'weight_decay': 0.01,
    'epochs': 80,
    'warmup_steps': 1000,
    'gradient_accumulation': 4,         # effective batch = 256
    'max_grad_norm': 1.0,
    'seed': 42,
    'gradient_checkpointing': False,
    'use_amp': True,
}

# Evaluation configuration
EVAL_CONFIG = {
    'eval_every': 5,
    'save_every': 5,
    'early_stopping_patience': 15,
}

# Experiment configurations
# NOTE: contrastive_loss is DISABLED by default to preserve biological variation
# batch_embedding is kept to capture batch-specific biases (can be removed post-hoc)
EXPERIMENTS = {
    'scMaizeExp': {
        'model_type': 'scMaizeExp',
        'use_func_embedding': False,
        'func_embed_path': None,
        'checkpoint_prefix': 'scMaizeExp',
        'use_batch_label': True,       # Safe: conditional bias addition
        'use_contrastive': False,      # Disabled: could remove biological variation
        'contrastive_weight': 0.0,     # Disabled
    },
    'scMaizeGO': {
        'model_type': 'scMaizeGO',
        'use_func_embedding': True,
        'func_embed_path': PATH_CONFIG['go_embedding_path'],
        'checkpoint_prefix': 'scMaizeGO',
        'use_batch_label': True,       # Safe: conditional bias addition
        'use_contrastive': False,      # Disabled: could remove biological variation
        'contrastive_weight': 0.0,     # Disabled
    },
}

def get_config():
    return {
        'PATH_CONFIG': PATH_CONFIG,
        'MODEL_CONFIG': MODEL_CONFIG,
        'DATA_CONFIG': DATA_CONFIG,
        'TRAINING_CONFIG': TRAINING_CONFIG,
        'EVAL_CONFIG': EVAL_CONFIG,
        'EXPERIMENTS': EXPERIMENTS,
    }
