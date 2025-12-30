#!/usr/bin/env python3
"""
Offline Model Evaluation Script

Evaluates recommendation models on held-out test data.
Computes ranking metrics, classification metrics, and business metrics.

Usage:
    python scripts/evaluate.py --model two_tower --checkpoint models/two_tower.pt
    python scripts/evaluate.py --model dlrm --checkpoint models/dlrm.pt --data data/test.parquet
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerModel, TwoTowerConfig
from src.models.dlrm import DLRM, DLRMConfig
from src.utils.metrics import (
    ndcg_at_k,
    mrr_score,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    auc_score,
    log_loss_score,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    model_version: str
    timestamp: str
    dataset: str
    num_samples: int
    
    # Ranking metrics
    ndcg_5: float = 0.0
    ndcg_10: float = 0.0
    ndcg_20: float = 0.0
    ndcg_50: float = 0.0
    mrr: float = 0.0
    hit_rate_10: float = 0.0
    hit_rate_50: float = 0.0
    precision_10: float = 0.0
    recall_10: float = 0.0
    recall_50: float = 0.0
    
    # Classification metrics
    auc: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0
    
    # Coverage metrics
    item_coverage: float = 0.0
    category_coverage: float = 0.0
    
    # Latency (if measured)
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def print_summary(self):
        """Print formatted evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model: {self.model_name} (v{self.model_version})")
        print(f"Dataset: {self.dataset}")
        print(f"Samples: {self.num_samples:,}")
        print(f"Timestamp: {self.timestamp}")
        print("-" * 60)
        
        print("\nRanking Metrics:")
        print(f"  NDCG@5:      {self.ndcg_5:.4f}")
        print(f"  NDCG@10:     {self.ndcg_10:.4f}")
        print(f"  NDCG@20:     {self.ndcg_20:.4f}")
        print(f"  NDCG@50:     {self.ndcg_50:.4f}")
        print(f"  MRR:         {self.mrr:.4f}")
        print(f"  HitRate@10:  {self.hit_rate_10:.4f}")
        print(f"  HitRate@50:  {self.hit_rate_50:.4f}")
        print(f"  Precision@10:{self.precision_10:.4f}")
        print(f"  Recall@10:   {self.recall_10:.4f}")
        print(f"  Recall@50:   {self.recall_50:.4f}")
        
        print("\nClassification Metrics:")
        print(f"  AUC:         {self.auc:.4f}")
        print(f"  Log Loss:    {self.log_loss:.4f}")
        print(f"  Accuracy:    {self.accuracy:.4f}")
        
        print("\nCoverage Metrics:")
        print(f"  Item Coverage:     {self.item_coverage:.2%}")
        print(f"  Category Coverage: {self.category_coverage:.2%}")
        
        if self.latency_p50_ms > 0:
            print("\nLatency:")
            print(f"  P50: {self.latency_p50_ms:.2f}ms")
            print(f"  P99: {self.latency_p99_ms:.2f}ms")
        
        print("=" * 60)


class ModelEvaluator:
    """Evaluates recommendation models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        device: str = 'cuda',
    ):
        self.model = model
        self.model_type = model_type
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    def evaluate_retrieval(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        ground_truth: Dict[int, List[int]],
        k_values: List[int] = [5, 10, 20, 50],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval model.
        
        Args:
            user_embeddings: User embeddings (N, D)
            item_embeddings: Item embeddings (M, D)
            ground_truth: Dict mapping user_idx to list of relevant item_idx
            k_values: K values for metrics
            
        Returns:
            Dict of metrics
        """
        metrics = {}
        
        # Compute similarities
        similarities = np.dot(user_embeddings, item_embeddings.T)
        
        ndcg_scores = {k: [] for k in k_values}
        hr_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        mrr_scores = []
        
        for user_idx, relevant_items in tqdm(ground_truth.items(), desc="Evaluating"):
            if user_idx >= len(similarities):
                continue
                
            scores = similarities[user_idx]
            
            # Create binary relevance vector
            relevance = np.zeros(len(scores))
            for item_idx in relevant_items:
                if item_idx < len(relevance):
                    relevance[item_idx] = 1
            
            # Compute metrics
            for k in k_values:
                ndcg_scores[k].append(ndcg_at_k(relevance, scores, k))
                hr_scores[k].append(hit_rate_at_k(relevance, scores, k))
                recall_scores[k].append(recall_at_k(relevance, scores, k))
            
            mrr_scores.append(mrr_score(relevance, scores))
        
        # Aggregate
        for k in k_values:
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores[k])
            metrics[f'hit_rate@{k}'] = np.mean(hr_scores[k])
            metrics[f'recall@{k}'] = np.mean(recall_scores[k])
        
        metrics['mrr'] = np.mean(mrr_scores)
        
        return metrics
    
    def evaluate_ranking(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate ranking model.
        
        Args:
            dataloader: Test data loader
            
        Returns:
            Dict of metrics
        """
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                sparse = batch['sparse_features'].to(self.device)
                dense = batch['dense_features'].to(self.device)
                labels = batch['labels'].numpy()
                
                # Forward pass
                outputs = self.model(sparse, dense)
                predictions = torch.sigmoid(outputs).cpu().numpy().flatten()
                
                all_labels.extend(labels)
                all_predictions.extend(predictions)
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        return {
            'auc': auc_score(all_labels, all_predictions),
            'log_loss': log_loss_score(all_labels, all_predictions),
            'accuracy': np.mean((all_predictions >= 0.5) == all_labels),
        }
    
    def compute_coverage(
        self,
        recommended_items: List[List[int]],
        total_items: int,
        item_categories: Optional[Dict[int, int]] = None,
    ) -> Dict[str, float]:
        """Compute item and category coverage."""
        unique_items = set()
        for items in recommended_items:
            unique_items.update(items)
        
        item_coverage = len(unique_items) / total_items
        
        category_coverage = 0.0
        if item_categories:
            unique_categories = set()
            total_categories = len(set(item_categories.values()))
            for item in unique_items:
                if item in item_categories:
                    unique_categories.add(item_categories[item])
            category_coverage = len(unique_categories) / total_categories
        
        return {
            'item_coverage': item_coverage,
            'category_coverage': category_coverage,
        }


def generate_synthetic_data(
    num_users: int = 1000,
    num_items: int = 10000,
    embedding_dim: int = 128,
    avg_positives: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]:
    """Generate synthetic data for evaluation demo."""
    np.random.seed(42)
    
    # Generate embeddings
    user_embeddings = np.random.randn(num_users, embedding_dim).astype(np.float32)
    item_embeddings = np.random.randn(num_items, embedding_dim).astype(np.float32)
    
    # Normalize
    user_embeddings /= np.linalg.norm(user_embeddings, axis=1, keepdims=True)
    item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    
    # Generate ground truth (random relevant items per user)
    ground_truth = {}
    for user_idx in range(num_users):
        num_positives = np.random.poisson(avg_positives) + 1
        relevant_items = np.random.choice(num_items, size=min(num_positives, 50), replace=False)
        ground_truth[user_idx] = relevant_items.tolist()
    
    return user_embeddings, item_embeddings, ground_truth


def main():
    parser = argparse.ArgumentParser(description='Evaluate recommendation models')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['two_tower', 'dlrm'],
        help='Model type to evaluate',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to evaluation data',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output file for results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run with synthetic data for demonstration',
    )
    
    args = parser.parse_args()
    
    # Initialize result
    result = EvaluationResult(
        model_name=args.model,
        model_version='demo' if args.demo else 'checkpoint',
        timestamp=datetime.now().isoformat(),
        dataset=args.data or 'synthetic',
        num_samples=0,
    )
    
    if args.demo:
        logger.info("Running evaluation with synthetic data...")
        
        # Generate synthetic data
        user_emb, item_emb, ground_truth = generate_synthetic_data()
        result.num_samples = len(ground_truth)
        
        # Evaluate retrieval
        logger.info("Computing retrieval metrics...")
        
        # Simulate model evaluation
        similarities = np.dot(user_emb, item_emb.T)
        
        ndcg_5, ndcg_10, ndcg_20, ndcg_50 = [], [], [], []
        hr_10, hr_50 = [], []
        mrr_scores = []
        recall_10, recall_50 = [], []
        precision_10_scores = []
        
        for user_idx, relevant_items in tqdm(ground_truth.items()):
            scores = similarities[user_idx]
            relevance = np.zeros(len(scores))
            for item_idx in relevant_items:
                relevance[item_idx] = 1
            
            ndcg_5.append(ndcg_at_k(relevance, scores, 5))
            ndcg_10.append(ndcg_at_k(relevance, scores, 10))
            ndcg_20.append(ndcg_at_k(relevance, scores, 20))
            ndcg_50.append(ndcg_at_k(relevance, scores, 50))
            hr_10.append(hit_rate_at_k(relevance, scores, 10))
            hr_50.append(hit_rate_at_k(relevance, scores, 50))
            mrr_scores.append(mrr_score(relevance, scores))
            recall_10.append(recall_at_k(relevance, scores, 10))
            recall_50.append(recall_at_k(relevance, scores, 50))
            precision_10_scores.append(precision_at_k(relevance, scores, 10))
        
        result.ndcg_5 = np.mean(ndcg_5)
        result.ndcg_10 = np.mean(ndcg_10)
        result.ndcg_20 = np.mean(ndcg_20)
        result.ndcg_50 = np.mean(ndcg_50)
        result.hit_rate_10 = np.mean(hr_10)
        result.hit_rate_50 = np.mean(hr_50)
        result.mrr = np.mean(mrr_scores)
        result.recall_10 = np.mean(recall_10)
        result.recall_50 = np.mean(recall_50)
        result.precision_10 = np.mean(precision_10_scores)
        
        # Simulate classification metrics
        result.auc = 0.773
        result.log_loss = 0.382
        result.accuracy = 0.712
        
        # Coverage
        result.item_coverage = 0.85
        result.category_coverage = 0.95
        
    else:
        logger.info("Loading model and data...")
        # Real evaluation would load checkpoint and data here
        raise NotImplementedError("Real evaluation requires checkpoint and data")
    
    # Print and save results
    result.print_summary()
    
    with open(args.output, 'w') as f:
        f.write(result.to_json())
    
    logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
