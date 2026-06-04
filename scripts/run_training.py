#!/usr/bin/env python3
"""
scMaize 训练脚本
================
- 支持 scMaizeExp 和 scMaizeGO 两个实验
- 支持断点恢复 (--resume)
- 支持 AMP 混合精度训练
- 支持 cosine LR scheduler with warmup
- 支持 Batch Label (可选)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import time
import gc
import sys
import json
import logging
from datetime import datetime
import math
import argparse
from typing import Dict, Any, Optional

# 添加脚本路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from configs import training_config as cfg
from model_scmaize import MGMLoss, create_model
from dataloader_scmaize import SingleCellDataset, collate_fn


def setup_logging(exp_name: str) -> logging.Logger:
    """设置日志"""
    log_dir = cfg.PATH_CONFIG['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{exp_name}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    for handler in logging.root.handlers:
        handler.flush = sys.stdout.flush

    return logging.getLogger(), log_file


def evaluate_model(model, loader, device, use_batch_label=False):
    """在验证集/测试集上评估模型，返回 MSE 和 Pearson"""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for b in loader:
            g = b['gene_ids'].to(device)
            e = b['expr_values'].to(device)
            t = b['mgm_target'].to(device)
            m = b['mgm_mask']
            b_label = b['batch_labels'].to(device) if use_batch_label else None
            log, _ = model(g, e, batch_labels=b_label)
            preds.extend(log[m].cpu().numpy())
            targets.extend(t[m].cpu().numpy())
    
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr
    mse = mean_squared_error(targets, preds)
    pear = pearsonr(preds, targets)[0]
    return mse, pear


def train_experiment(
    exp_name: str,
    exp_config: Dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    expr_data: np.ndarray,
    batch_labels: np.ndarray,
    gene_list: list,
    n_genes: int,
    cell_types: Optional[np.ndarray],
    logger: logging.Logger,
    device: str = 'cuda',
    resume_from: Optional[str] = None,
    start_epoch: int = 0,
    resume_start_epoch: int = 0,
    total_epochs: int = 80,
    loaded_history: Optional[Dict] = None
):
    """训练单个实验"""

    model_cfg = cfg.MODEL_CONFIG
    train_cfg = cfg.TRAINING_CONFIG

    # 数据集
    seq_len = cfg.DATA_CONFIG['seq_len']
    mask_ratio = cfg.DATA_CONFIG['mask_ratio']

    # 加载 batch_labels (如果存在)
    batch_labels_data = batch_labels if exp_config.get('use_batch_label', False) else None

    train_ds = SingleCellDataset(
        expr_data[train_idx], gene_list, seq_len, mask_ratio,
        n_genes_total=n_genes,
        batch_labels=batch_labels_data[train_idx] if batch_labels_data is not None else None
    )
    val_ds = SingleCellDataset(
        expr_data[val_idx], gene_list, seq_len, mask_ratio,
        n_genes_total=n_genes,
        batch_labels=batch_labels_data[val_idx] if batch_labels_data is not None else None
    )
    test_ds = SingleCellDataset(
        expr_data[test_idx], gene_list, seq_len, mask_ratio,
        n_genes_total=n_genes,
        batch_labels=batch_labels_data[test_idx] if batch_labels_data is not None else None
    )

    batch_size = train_cfg['batch_size']
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    logger.info(f"实验 {exp_name}: 训练 {len(train_ds)}, 验证 {len(val_ds)}, 测试 {len(test_ds)}")
    print(f"每轮迭代次数: {len(train_loader)}", flush=True)

    # 模型
    model = create_model(
        model_type=exp_config['model_type'],
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['hidden_dim'],
        n_heads=model_cfg['num_heads'],
        n_layers=model_cfg['num_layers'],
        d_ff=model_cfg['hidden_dim'] * model_cfg['mlp_expansion'],
        dropout=model_cfg['dropout'],
        use_batch_label=exp_config.get('use_batch_label', True),
        use_contrastive=exp_config.get('use_contrastive', False),
        temperature=0.1,
        func_embed_path=exp_config.get('func_embed_path'),
        use_gradient_checkpointing=train_cfg['gradient_checkpointing'],
        device=device
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"参数: {n_params:.1f}M")
    print(f"参数: {n_params:.1f}M", flush=True)

    # AMP
    scaler = torch.amp.GradScaler('cuda')

    # Optimizer
    opt = optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )

    # MGM Loss
    mgm_fn = MGMLoss()

    # Scheduler
    steps_per_epoch = len(train_loader)
    warmup_steps = train_cfg['warmup_steps']
    grad_accum = train_cfg['gradient_accumulation']
    scheduler_steps_per_epoch = (steps_per_epoch + grad_accum - 1) // grad_accum
    total_steps = scheduler_steps_per_epoch * total_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 历史记录
    history = loaded_history if loaded_history is not None else {
        'train_loss': [], 'val_mse': [], 'val_pear': [],
        'val_mse_no_batch': [], 'val_pear_no_batch': [],
        'test_mse': [], 'test_pear': []
    }

    best_mse = float('inf')
    best_state = None
    best_test_mse = float('inf')
    best_test_pear = 0.0

    if history.get('best_val_mse') is not None:
        best_mse = history['best_val_mse']
        best_test_mse = history.get('best_test_mse', float('inf'))
        best_test_pear = history.get('best_test_pear', 0.0)
        if history.get('best_state_dict') is not None:
            best_state = history['best_state_dict']

    epochs_without_improvement = 0
    epochs = total_epochs
    eval_every = cfg.EVAL_CONFIG['eval_every']
    save_every = cfg.EVAL_CONFIG['save_every']
    early_stopping_patience = cfg.EVAL_CONFIG.get('early_stopping_patience', 15)

    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr

    # 初始评估
    if start_epoch == 0:
        logger.info("=" * 50)
        logger.info("评估初始模型 (Epoch 0 - 训练前 baseline)")
        print("评估初始模型...", flush=True)
        
        val_mse_init, val_pear_init = evaluate_model(model, val_loader, device, exp_config.get('use_batch_label', False))
        
        history['train_loss'].append(float('inf'))
        history['val_mse'].append(float(val_mse_init))
        history['val_pear'].append(float(val_pear_init))
        
        logger.info(f"初始验证集: MSE={val_mse_init:.4f}, Pearson={val_pear_init:.4f}")
        logger.info("=" * 50)

    train_ds.regenerate_gene_indices()
    train_ds.regenerate_mask()

    for epoch in range(start_epoch, epochs):
        train_ds.regenerate_gene_indices()
        train_ds.regenerate_mask()

        t0 = time.time()
        model.train()

        epoch_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            g = batch['gene_ids'].to(device)
            e = batch['expr_values'].to(device)
            t = batch['mgm_target'].to(device)
            m = batch['mgm_mask'].float().to(device)
            b = batch['batch_labels'].to(device) if exp_config.get('use_batch_label', False) else None

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                log, _ = model(g, e, batch_labels=b)
                loss = mgm_fn(log, t, m)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['max_grad_norm'])
                scaler.step(opt)
                scaler.update()
                scheduler.step()

            epoch_loss += loss.item() * grad_accum
            n_batches += 1

            if (batch_idx + 1) % 500 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item()*grad_accum:.4f}, LR: {current_lr:.2e}", flush=True)

        avg_loss = epoch_loss / n_batches
        et = time.time() - t0
        history['train_loss'].append(float(avg_loss))

        # 评估
        if (epoch + 1) % eval_every == 0:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for b in val_loader:
                    g = b['gene_ids'].to(device)
                    e = b['expr_values'].to(device)
                    t = b['mgm_target'].to(device)
                    m = b['mgm_mask']
                    b_label = b['batch_labels'].to(device) if exp_config.get('use_batch_label', False) else None
                    log, _ = model(g, e, batch_labels=b_label)
                    preds.extend(log[m].cpu().numpy())
                    targets.extend(t[m].cpu().numpy())

            val_mse = mean_squared_error(targets, preds)
            val_mse = mean_squared_error(targets, preds)
            val_pear = pearsonr(preds, targets)[0]

            # 公平对比：不带 batch label
            preds_nb, targets_nb = [], []
            with torch.no_grad():
                for b in val_loader:
                    g = b['gene_ids'].to(device)
                    e = b['expr_values'].to(device)
                    t = b['mgm_target'].to(device)
                    m = b['mgm_mask']
                    log, _ = model(g, e, batch_labels=None)
                    preds_nb.extend(log[m].cpu().numpy())
                    targets_nb.extend(t[m].cpu().numpy())
            val_mse_no_batch = mean_squared_error(targets_nb, preds_nb)
            val_pear_no_batch = pearsonr(preds_nb, targets_nb)[0]

            history['val_mse'].append(float(val_mse))
            history['val_pear'].append(float(val_pear))
            history['val_mse_no_batch'].append(float(val_mse_no_batch))
            history['val_pear_no_batch'].append(float(val_pear_no_batch))

            logger.info(f"Epoch {epoch+1}/{epochs} | {et:.1f}s | "
                        f"Loss: {avg_loss:.4f} | Val MSE: {val_mse:.4f}(nb:{val_mse_no_batch:.4f}) | "
                        f"Val Pearson: {val_pear:.4f}(nb:{val_pear_no_batch:.4f})")
            print(f"Epoch {epoch+1}/{epochs} | {et:.1f}s | "
                  f"Loss: {avg_loss:.4f} | Val MSE: {val_mse:.4f}(nb:{val_mse_no_batch:.4f}) | "
                  f"Val Pearson: {val_pear:.4f}(nb:{val_pear_no_batch:.4f})", flush=True)

            if val_mse < best_mse:
                best_mse = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                history['best_val_mse'] = float(best_mse)

                # 评估测试集
                test_preds, test_targets = [], []
                with torch.no_grad():
                    for b in test_loader:
                        g = b['gene_ids'].to(device)
                        e = b['expr_values'].to(device)
                        t = b['mgm_target'].to(device)
                        m = b['mgm_mask']
                        b_label = b['batch_labels'].to(device) if exp_config.get('use_batch_label', False) else None
                        log, _ = model(g, e, batch_labels=b_label)
                        test_preds.extend(log[m].cpu().numpy())
                        test_targets.extend(t[m].cpu().numpy())

                best_test_mse = mean_squared_error(test_targets, test_preds)
                best_test_pear = pearsonr(test_preds, test_targets)[0]
                history['best_test_mse'] = float(best_test_mse)
                history['best_test_pear'] = float(best_test_pear)

                logger.info(f"  -> 新最佳! 测试集: MSE: {best_test_mse:.4f}, "
                            f"Pearson: {best_test_pear:.4f}")
                print(f"  -> 新最佳! 测试集: MSE: {best_test_mse:.4f}, "
                      f"Pearson: {best_test_pear:.4f}", flush=True)
                
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logger.info(f"  未改善: 已连续 {epochs_without_improvement}/{early_stopping_patience} 个 epoch 无改善")
                
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"早停: 连续 {early_stopping_patience} 个 epoch 无改善")
                    break

        else:
            logger.info(f"Epoch {epoch+1}/{epochs} | {et:.1f}s | Loss: {avg_loss:.4f}")
            print(f"Epoch {epoch+1}/{epochs} | {et:.1f}s | Loss: {avg_loss:.4f}", flush=True)

        # 保存 checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_dir = cfg.PATH_CONFIG['checkpoint_dir']
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ckpt_dir, f"{exp_config['checkpoint_prefix']}_epoch{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'history': history,
                'train_ds_seed': train_ds.get_seed(),
                'learning_rate': opt.param_groups[0]['lr'],
            }, ckpt_path)
            logger.info(f"Checkpoint 保存: {ckpt_path}")

        gc.collect()
        torch.cuda.empty_cache()

    # 保存最终最佳模型
    ckpt_dir = cfg.PATH_CONFIG['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(
        ckpt_dir, f"{exp_config['checkpoint_prefix']}_best.pt"
    )
    if best_state is not None:
        best_checkpoint = {
            'model_state_dict': best_state,
            'learning_rate': opt.param_groups[0]['lr'],
        }
        torch.save(best_checkpoint, best_path)
        logger.info(f"最佳模型保存: {best_path}")

    logger.info(f"最佳验证MSE: {best_mse:.4f}, 测试MSE: {best_test_mse:.4f}, Pearson: {best_test_pear:.4f}")
    print(f"最佳验证MSE: {best_mse:.4f}, 测试MSE: {best_test_mse:.4f}, Pearson: {best_test_pear:.4f}", flush=True)

    return {
        'val_mse': float(best_mse),
        'test_mse': float(best_test_mse),
        'test_pearson': float(best_test_pear),
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(description='scMaize 训练')
    parser.add_argument('--exp', type=str, choices=['scMaizeExp', 'scMaizeGO'],
                        help='实验类型: scMaizeExp 或 scMaizeGO')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复的路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda 或 cpu')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练总 epoch 数')
    parser.add_argument('--flash-sdp', action='store_true',
                        help='启用 PyTorch SDPA 加速 (需要 PyTorch 2.0+)')
    args = parser.parse_args()

    # 固定随机种子，确保可复现性
    seed = cfg.TRAINING_CONFIG.get('seed', 42)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 60, flush=True)
    print("scMaize 训练 (V2 优化版)", flush=True)
    print("=" * 60, flush=True)

    # 启用 PyTorch SDPA 加速
    if args.flash_sdp and torch.cuda.is_available():
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("已启用 PyTorch SDPA 加速", flush=True)
        else:
            print("警告: PyTorch 版本不支持 SDPA，请升级到 2.0+", flush=True)
    else:
        print("使用标准 Attention", flush=True)

    base_dir = PROJECT_DIR
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}", flush=True)

    torch.backends.cudnn.benchmark = True

    # 加载预处理数据
    print("\n加载数据...", flush=True)
    data_path = cfg.PATH_CONFIG['preprocessed_data']
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    n_genes = len(data['gene_list'])
    expr_data = data['expr_continuous']
    cell_types = data['cell_types']
    gene_list = data['gene_list']
    batch_labels = data.get('batch_labels', None)
    n_samples = len(expr_data)

    print(f"基因数: {n_genes}, 总样本: {n_samples}", flush=True)
    if batch_labels is not None:
        print(f"批次标签: {len(np.unique(batch_labels))} batches", flush=True)

    # 数据划分 (复用预处理数据中的划分)
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    print(f"训练: {len(train_idx)}, 验证: {len(val_idx)}, 测试: {len(test_idx)}", flush=True)

    # 确定实验配置
    if args.exp == 'scMaizeExp':
        exp_name = 'scMaizeExp'
        exp_config = cfg.EXPERIMENTS['scMaizeExp']
    elif args.exp == 'scMaizeGO':
        exp_name = 'scMaizeGO'
        exp_config = cfg.EXPERIMENTS['scMaizeGO']
    else:
        raise ValueError(f"未知的实验类型: {args.exp}")

    # 设置日志
    logger, log_file = setup_logging(exp_name)
    logger.info(f"实验: {exp_name}")
    logger.info(f"配置: {exp_config}")
    print(f"实验: {exp_name}", flush=True)

    # 恢复相关信息
    resume_from = None
    start_epoch = 0
    loaded_history = None

    if args.resume is not None:
        resume_from = args.resume
        checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)
        start_epoch = checkpoint.get('epoch', 0)
        loaded_history = checkpoint.get('history', None)
        logger.info(f"将从 epoch {start_epoch} 恢复训练")

    total_epochs = args.epochs if args.epochs is not None else cfg.TRAINING_CONFIG['epochs']
    if args.epochs is not None:
        logger.info(f"训练总 epoch 数: {total_epochs}")

    # 训练
    result = train_experiment(
        exp_name=exp_name,
        exp_config=exp_config,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        expr_data=expr_data,
        batch_labels=batch_labels,
        gene_list=gene_list,
        n_genes=n_genes,
        cell_types=None,

        logger=logger,
        device=device,
        resume_from=resume_from,
        start_epoch=start_epoch,
        loaded_history=loaded_history,
        resume_start_epoch=start_epoch,
        total_epochs=total_epochs
    )

    # 保存结果
    exp_dir = cfg.PATH_CONFIG['experiment_dir']
    os.makedirs(exp_dir, exist_ok=True)
    results_path = os.path.join(exp_dir, f'results_{exp_name}.json')
    with open(results_path, 'w') as f:
        json.dump({
            'exp_name': exp_name,
            'config': exp_config,
            'result': {
                'val_mse': result['val_mse'],
                'test_mse': result['test_mse'],
                'test_pearson': result['test_pearson'],
            }
        }, f, indent=2)

    history_path = os.path.join(exp_dir, f'history_{exp_name}.json')
    with open(history_path, 'w') as f:
        hist_clean = {k: v for k, v in result['history'].items() if k != 'best_state_dict'}
        json.dump(hist_clean, f, indent=2)

    print(f"\n结果保存: {results_path}", flush=True)
    print(f"历史保存: {history_path}", flush=True)
    print(f"日志文件: {log_file}", flush=True)


if __name__ == "__main__":
    main()
