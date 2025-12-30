#!/usr/bin/env python3
"""
Training Script for Real-Time Personalization Engine

Supports training Two-Tower retrieval and DLRM ranking models with:
- Distributed training across multiple GPUs
- Mixed precision training (FP16/BF16)
- MLflow experiment tracking
- Checkpoint management
- Hyperparameter optimization with Optuna

Usage:
    # Train Two-Tower model
    python scripts/train.py --model two_tower --config configs/model_config.yaml

    # Train DLRM model with distributed training
    python scripts/train.py --model dlrm --distributed --gpus 8

    # Hyperparameter optimization
    python scripts/train.py --model two_tower --optimize --trials 100
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerConfig, TwoTowerModel
from src.models.dlrm import DLRMConfig, DLRM, MultiTaskDLRM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Classes
# =============================================================================

class RecommendationDataset(Dataset):
    """Dataset for recommendation model training."""
    
    def __init__(
        self,
        data_path: str,
        model_type: str = "two_tower",
        max_history_length: int = 50,
        split: str = "train"
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data directory
            model_type: "two_tower" or "dlrm"
            max_history_length: Maximum user history length
            split: "train", "val", or "test"
        """
        self.data_path = Path(data_path)
        self.model_type = model_type
        self.max_history_length = max_history_length
        self.split = split
        
        # Load data (placeholder - would load from parquet/tfrecord in production)
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess data."""
        # Generate synthetic data for demonstration
        # In production, load from parquet files with NVTabular
        np.random.seed(42 if self.split == "train" else 43)
        
        n_samples = 1_000_000 if self.split == "train" else 100_000
        
        self.user_ids = np.random.randint(0, 100_000, n_samples)
        self.item_ids = np.random.randint(0, 1_000_000, n_samples)
        self.labels = np.random.binomial(1, 0.05, n_samples).astype(np.float32)
        
        # User categorical features
        self.user_age_bucket = np.random.randint(0, 10, n_samples)
        self.user_gender = np.random.randint(0, 3, n_samples)
        self.user_country = np.random.randint(0, 50, n_samples)
        
        # Item categorical features
        self.item_category = np.random.randint(0, 1000, n_samples)
        self.item_brand = np.random.randint(0, 5000, n_samples)
        
        # Dense features
        self.user_dense = np.random.randn(n_samples, 10).astype(np.float32)
        self.item_dense = np.random.randn(n_samples, 8).astype(np.float32)
        
        # User history (for Two-Tower)
        self.user_history = np.random.randint(
            0, 1_000_000, (n_samples, self.max_history_length)
        )
        self.history_mask = np.random.binomial(
            1, 0.7, (n_samples, self.max_history_length)
        ).astype(np.float32)
        
        logger.info(f"Loaded {n_samples:,} samples for {self.split} split")
    
    def __len__(self) -> int:
        return len(self.user_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.model_type == "two_tower":
            return {
                'user_categorical': torch.tensor([
                    self.user_ids[idx],
                    self.user_age_bucket[idx],
                    self.user_gender[idx],
                    self.user_country[idx]
                ], dtype=torch.long),
                'user_dense': torch.tensor(self.user_dense[idx], dtype=torch.float32),
                'user_history': torch.tensor(self.user_history[idx], dtype=torch.long),
                'history_mask': torch.tensor(self.history_mask[idx], dtype=torch.float32),
                'item_categorical': torch.tensor([
                    self.item_ids[idx],
                    self.item_category[idx],
                    self.item_brand[idx]
                ], dtype=torch.long),
                'item_dense': torch.tensor(self.item_dense[idx], dtype=torch.float32),
                'label': torch.tensor(self.labels[idx], dtype=torch.float32)
            }
        else:  # DLRM
            # Combine all sparse features
            sparse_features = torch.tensor([
                self.user_ids[idx],
                self.user_age_bucket[idx],
                self.user_gender[idx],
                self.user_country[idx],
                self.item_ids[idx],
                self.item_category[idx],
                self.item_brand[idx]
            ], dtype=torch.long)
            
            # Combine all dense features
            dense_features = torch.cat([
                torch.tensor(self.user_dense[idx], dtype=torch.float32),
                torch.tensor(self.item_dense[idx], dtype=torch.float32)
            ])
            
            return {
                'sparse_features': sparse_features,
                'dense_features': dense_features,
                'label': torch.tensor(self.labels[idx], dtype=torch.float32)
            }


# =============================================================================
# Training Utilities
# =============================================================================

class MetricTracker:
    """Track and aggregate training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, name: str, value: float, count: int = 1):
        if name not in self.metrics:
            self.metrics[name] = 0.0
            self.counts[name] = 0
        self.metrics[name] += value * count
        self.counts[name] += count
    
    def get(self, name: str) -> float:
        if name not in self.metrics or self.counts[name] == 0:
            return 0.0
        return self.metrics[name] / self.counts[name]
    
    def get_all(self) -> Dict[str, float]:
        return {name: self.get(name) for name in self.metrics}


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best_only: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_metric = float('inf')
        self.checkpoints = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metric: float,
        config: Dict[str, Any]
    ) -> Optional[str]:
        """Save checkpoint if conditions are met."""
        if self.save_best_only and metric >= self.best_metric:
            return None
        
        self.best_metric = min(self.best_metric, metric)
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}_{metric:.4f}.pt"
        
        # Handle DDP wrapped models
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metric': metric,
            'config': config
        }, checkpoint_path)
        
        self.checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        latest = checkpoints[-1]
        logger.info(f"Loading checkpoint: {latest}")
        return torch.load(latest)


# =============================================================================
# Training Functions
# =============================================================================

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    model_type: str,
    accumulation_steps: int = 1
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metrics = MetricTracker()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with autocast():
            if model_type == "two_tower":
                # Two-Tower forward pass
                user_embedding = model.user_tower(
                    batch['user_categorical'],
                    batch['user_dense'],
                    batch['user_history'],
                    batch['history_mask']
                )
                item_embedding = model.item_tower(
                    batch['item_categorical'],
                    batch['item_dense']
                )
                
                # Compute loss with in-batch negatives
                logits = torch.matmul(user_embedding, item_embedding.t()) / model.config.temperature
                labels = torch.arange(logits.size(0), device=device)
                loss = nn.CrossEntropyLoss()(logits, labels)
                
            else:  # DLRM
                output = model(batch['sparse_features'], batch['dense_features'])
                loss = nn.BCEWithLogitsLoss()(output.squeeze(), batch['label'])
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update metrics
        metrics.update('loss', loss.item() * accumulation_steps, batch['label'].size(0))
        
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {metrics.get('loss'):.4f}")
    
    return metrics.get_all()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    metrics = MetricTracker()
    
    all_predictions = []
    all_labels = []
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if model_type == "two_tower":
            user_embedding = model.user_tower(
                batch['user_categorical'],
                batch['user_dense'],
                batch['user_history'],
                batch['history_mask']
            )
            item_embedding = model.item_tower(
                batch['item_categorical'],
                batch['item_dense']
            )
            predictions = torch.sum(user_embedding * item_embedding, dim=1)
        else:
            predictions = model(batch['sparse_features'], batch['dense_features']).squeeze()
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch['label'].cpu().numpy())
    
    # Compute metrics
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # AUC
    try:
        from sklearn.metrics import roc_auc_score, log_loss
        auc = roc_auc_score(labels, predictions)
        logloss = log_loss(labels, torch.sigmoid(torch.tensor(predictions)).numpy())
        metrics.update('auc', auc)
        metrics.update('logloss', logloss)
    except Exception as e:
        logger.warning(f"Could not compute AUC: {e}")
    
    return metrics.get_all()


def train(
    config: Dict[str, Any],
    model_type: str = "two_tower",
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
):
    """Main training function."""
    # Setup device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    if model_type == "two_tower":
        model_config = TwoTowerConfig(**config['models']['two_tower'])
        model = TwoTowerModel(model_config)
    else:
        model_config = DLRMConfig(**config['models']['dlrm'])
        model = DLRM(model_config)
    
    model = model.to(device)
    
    # Distributed wrapper
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Create datasets and dataloaders
    train_dataset = RecommendationDataset(
        config['training']['data_path'],
        model_type=model_type,
        split="train"
    )
    val_dataset = RecommendationDataset(
        config['training']['data_path'],
        model_type=model_type,
        split="val"
    )
    
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['training'].get('num_workers', 8),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'] * 2,
        shuffle=False,
        num_workers=config['training'].get('num_workers', 8),
        pin_memory=True
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        epochs=config['training']['epochs'],
        steps_per_epoch=len(train_loader)
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        config['training'].get('checkpoint_dir', 'checkpoints'),
        max_checkpoints=5,
        save_best_only=True
    )
    
    # MLflow tracking
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    if mlflow_uri and rank == 0:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(f"recommendation_{model_type}")
            mlflow.start_run()
            mlflow.log_params({
                'model_type': model_type,
                'batch_size': config['training']['batch_size'],
                'learning_rate': config['training']['learning_rate'],
                'epochs': config['training']['epochs']
            })
        except Exception as e:
            logger.warning(f"Could not initialize MLflow: {e}")
    
    # Training loop
    best_auc = 0.0
    
    for epoch in range(config['training']['epochs']):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device,
            model_type, accumulation_steps=config['training'].get('accumulation_steps', 1)
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device, model_type)
        
        # Step scheduler
        scheduler.step()
        
        # Log metrics
        if rank == 0:
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Val AUC: {val_metrics.get('auc', 0):.4f}, "
                       f"Val LogLoss: {val_metrics.get('logloss', 0):.4f}")
            
            # Save checkpoint
            metric_to_track = val_metrics.get('logloss', train_metrics['loss'])
            checkpoint_manager.save(
                model, optimizer, scheduler, epoch,
                metric_to_track, config
            )
            
            # MLflow logging
            if mlflow_uri:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        'train_loss': train_metrics['loss'],
                        'val_auc': val_metrics.get('auc', 0),
                        'val_logloss': val_metrics.get('logloss', 0)
                    }, step=epoch)
                except Exception:
                    pass
            
            if val_metrics.get('auc', 0) > best_auc:
                best_auc = val_metrics['auc']
                logger.info(f"New best AUC: {best_auc:.4f}")
    
    # Cleanup
    if mlflow_uri and rank == 0:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass
    
    if distributed:
        cleanup_distributed()
    
    return best_auc


def run_hyperparameter_optimization(
    config: Dict[str, Any],
    model_type: str,
    n_trials: int = 100
):
    """Run hyperparameter optimization with Optuna."""
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        return
    
    def objective(trial):
        # Suggest hyperparameters
        config['training']['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True
        )
        config['training']['batch_size'] = trial.suggest_categorical(
            'batch_size', [1024, 2048, 4096, 8192]
        )
        
        if model_type == "two_tower":
            config['models']['two_tower']['embedding_dim'] = trial.suggest_categorical(
                'embedding_dim', [64, 128, 256]
            )
            config['models']['two_tower']['temperature'] = trial.suggest_float(
                'temperature', 0.01, 0.1
            )
        else:
            config['models']['dlrm']['embedding_dim'] = trial.suggest_categorical(
                'embedding_dim', [32, 64, 128]
            )
        
        # Train and return metric
        config['training']['epochs'] = 3  # Quick training for HPO
        auc = train(config, model_type=model_type)
        
        return auc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best trial: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")
    
    return study.best_trial


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument(
        "--model", type=str, default="two_tower",
        choices=["two_tower", "dlrm"],
        help="Model type to train"
    )
    parser.add_argument(
        "--config", type=str, default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed training"
    )
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="Number of GPUs for distributed training"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run hyperparameter optimization"
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Training {args.model} model")
    logger.info(f"Config: {args.config}")
    
    if args.optimize:
        run_hyperparameter_optimization(config, args.model, args.trials)
    elif args.distributed:
        import torch.multiprocessing as mp
        mp.spawn(
            train,
            args=(config, args.model, True, args.gpus),
            nprocs=args.gpus,
            join=True
        )
    else:
        train(config, model_type=args.model)


if __name__ == "__main__":
    main()
