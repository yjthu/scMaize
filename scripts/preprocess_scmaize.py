#!/usr/bin/env python3
"""
scMaize 数据预处理脚本
=====================
将 h5ad 文件转换为模型训练所需的格式

与原训练保持一致:
- 15,000 个高变基因 (与 GO embedding 一致)
- 数据划分复用 split_info.json
- 新增批次标签
"""

import numpy as np
import pickle
import h5py
import anndata as ad
import os
import json
import gc
from scipy import sparse

# 配置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# 源文件路径
H5AD_PATH = "/data/YanJ/PROJECT/OmicVerse/Maize/maize_project_final/maize_atlas_annotated.v8.h5ad"
GO_EMBEDDING_PATH = "/data/YanJ/PROJECT/OmicVerse/Maize/maize_project_final/go_embedding_exploration/05_results/embedding_matrices/improved/M3_improved_gene_embedding.pkl"

# 原训练数据 (用于获取基因列表和划分)
ORIGINAL_DATA_PATH = "/data/YanJ/PROJECT/OmicVerse/Maize/maize_project_final/training_15k/preprocessed_data_15k.pkl"
SPLIT_PATH = "/data/YanJ/PROJECT/OmicVerse/Maize/maize_project_final/training_15k/experiments/split_info.json"

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "preprocessed_data.pkl")


def load_gene_list():
    """从原训练数据加载基因列表 (与 GO embedding 一致)"""
    print("[1/6] 加载基因列表...")
    with open(ORIGINAL_DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    gene_list = data['gene_list']
    print(f"  基因数: {len(gene_list)}")
    print(f"  前5个: {gene_list[:5]}")
    return gene_list


def load_go_embedding():
    """验证 GO embedding 基因列表"""
    print("\n[2/6] 验证 GO embedding...")
    with open(GO_EMBEDDING_PATH, 'rb') as f:
        go_data = pickle.load(f)
    go_genes = go_data['genes']
    print(f"  GO基因数: {len(go_genes)}")
    print(f"  前5个: {go_genes[:5]}")
    return go_data


def load_data_split():
    """加载原训练数据划分"""
    print("\n[3/6] 加载数据划分...")
    with open(SPLIT_PATH, 'r') as f:
        split = json.load(f)
    print(f"  Train: {len(split['train_idx']):,}")
    print(f"  Val: {len(split['val_idx']):,}")
    print(f"  Test: {len(split['test_idx']):,}")
    return split


def process_h5ad(gene_list):
    """处理 h5ad 文件，提取表达矩阵和批次信息"""
    print("\n[4/6] 处理 h5ad 文件...")
    print(f"  读取: {H5AD_PATH}")
    
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"  细胞数: {adata.n_obs:,}")
    print(f"  基因数: {adata.n_vars:,}")
    
    # 构建基因名到索引的映射
    h5ad_genes = adata.var.index.tolist()
    gene_to_idx = {g: i for i, g in enumerate(h5ad_genes)}
    
    # 找到目标基因在 h5ad 中的索引
    gene_indices = []
    missing_genes = []
    for i, gene in enumerate(gene_list):
        if gene in gene_to_idx:
            gene_indices.append(gene_to_idx[gene])
        else:
            missing_genes.append(gene)
    
    if missing_genes:
        print(f"  ⚠️ 警告: {len(missing_genes)} 个基因未在 h5ad 中找到")
        print(f"     前5个缺失: {missing_genes[:5]}")
    
    print(f"  找到 {len(gene_indices)}/{len(gene_list)} 个基因")
    gene_indices = np.array(gene_indices, dtype=np.int64)
    
    # 提取表达矩阵 - 使用 layers/counts (原始counts，未log)
    # 注意: V8 h5ad的 adata.X 已经是log-transformed，不能再用log1p
    # 必须使用 layers/counts 然后再 log1p，与V1保持一致
    print("  提取表达矩阵 (从 layers/counts)...")
    
    # 获取 counts 数据
    counts_data = adata.layers['counts']
    
    # 选取特定基因列
    if sparse.issparse(counts_data):
        X_dense = counts_data[:, gene_indices].toarray()
    else:
        X_dense = counts_data[:, gene_indices]
    
    # Log1p 转换 (原始counts -> log1p)
    X_log = np.log1p(X_dense).astype(np.float32)
    
    print(f"  表达矩阵 shape: {X_log.shape}")
    print(f"  值范围: [{X_log.min():.4f}, {X_log.max():.4f}]")
    print(f"  均值: {X_log.mean():.4f}")
    
    # 提取批次信息
    print("  提取批次信息...")
    batch_labels = adata.obs['batch'].values
    batch_to_int = {b: i for i, b in enumerate(np.unique(batch_labels))}
    batch_labels_int = np.array([batch_to_int[b] for b in batch_labels])
    
    print(f"  批次数量: {len(batch_to_int)}")
    print(f"  批次分布 (前10):")
    unique, counts = np.unique(batch_labels, return_counts=True)
    for b, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"    {b}: {c:,}")
    
    # 提取细胞类型
    cell_types = adata.obs['cell_type'].values
    
    del adata, X_dense
    gc.collect()
    
    return X_log, batch_labels_int, cell_types, gene_indices


def save_preprocessed_data(gene_list, expr_matrix, batch_labels, cell_types, split):
    """保存预处理后的数据"""
    print("\n[5/6] 保存结果...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 构建输出数据
    output_data = {
        'gene_list': gene_list,
        'expr_continuous': expr_matrix,
        'batch_labels': batch_labels,
        'cell_types': cell_types,
        'train_idx': np.array(split['train_idx']),
        'val_idx': np.array(split['val_idx']),
        'test_idx': np.array(split['test_idx']),
    }
    
    # 保存
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(output_data, f, protocol=4)
    
    file_size = os.path.getsize(OUTPUT_PATH) / 1024**3
    print(f"  保存到: {OUTPUT_PATH}")
    print(f"  文件大小: {file_size:.2f} GB")
    
    return OUTPUT_PATH


def verify_output():
    """验证输出"""
    print("\n[6/6] 验证结果...")
    
    with open(OUTPUT_PATH, 'rb') as f:
        data = pickle.load(f)
    
    print("  字段:")
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: {v.shape} {v.dtype}")
        elif isinstance(v, list):
            print(f"    {k}: list[{len(v)}]")
        else:
            print(f"    {k}: {type(v)}")
    
    # 验证与 GO embedding 的一致性
    with open(GO_EMBEDDING_PATH, 'rb') as f:
        go_data = pickle.load(f)
    
    if data['gene_list'] == go_data['genes']:
        print("  ✅ 基因列表与 GO embedding 一致")
    else:
        print("  ❌ 基因列表与 GO embedding 不一致!")
    
    # 验证数据划分
    total = len(data['train_idx']) + len(data['val_idx']) + len(data['test_idx'])
    print(f"  ✅ 数据划分: {len(data['train_idx']):,} / {len(data['val_idx']):,} / {len(data['test_idx']):,} (总计 {total:,})")
    
    # 验证批次标签
    print(f"  ✅ 批次标签: {len(np.unique(data['batch_labels']))} batches")


def main():
    print("=" * 60)
    print("scMaize 数据预处理")
    print("=" * 60)
    
    # 1. 加载基因列表
    gene_list = load_gene_list()
    
    # 2. 验证 GO embedding
    go_data = load_go_embedding()
    
    # 验证基因列表一致性
    if gene_list != go_data['genes']:
        raise ValueError("基因列表与 GO embedding 不一致!")
    print("  ✅ 基因列表验证通过")
    
    # 3. 加载数据划分
    split = load_data_split()
    
    # 4. 处理 h5ad
    expr_matrix, batch_labels, cell_types, gene_indices = process_h5ad(gene_list)
    
    # 5. 保存
    save_preprocessed_data(gene_list, expr_matrix, batch_labels, cell_types, split)
    
    # 6. 验证
    verify_output()
    
    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
