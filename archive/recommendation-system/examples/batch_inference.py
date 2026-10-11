#!/usr/bin/env python3
"""
Batch Inference Example

This example demonstrates how to:
1. Process recommendations for multiple users in batch
2. Use GPU acceleration for large-scale inference
3. Export results to various formats

Usage:
    python examples/batch_inference.py --input users.csv --output recommendations.parquet
    python examples/batch_inference.py --user-ids 1,2,3,4,5 --output results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.models.two_tower import TwoTowerModel
from src.serving.retrieval import RetrievalService
from src.serving.ranking import RankingService
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BatchRecommender:
    """Batch recommendation processor for large-scale inference."""
    
    def __init__(
        self,
        model_path: str = "models/checkpoints/two_tower/best.pt",
        config_path: str = "configs/model_config.yaml",
        device: str = "auto"
    ):
        """
        Initialize batch recommender.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.config = load_config(config_path)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = TwoTowerModel.load(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize services
        self.retrieval_service = RetrievalService(self.model, self.config)
        self.ranking_service = RankingService(self.model, self.config)
    
    def recommend_batch(
        self,
        user_ids: List[int],
        num_recommendations: int = 10,
        batch_size: int = 64,
        show_progress: bool = True
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            num_recommendations: Number of recommendations per user
            batch_size: Processing batch size
            show_progress: Whether to show progress
            
        Returns:
            Dictionary mapping user_id to list of recommendations
        """
        results = {}
        total_users = len(user_ids)
        
        start_time = time.time()
        
        # Process in batches
        for i in range(0, total_users, batch_size):
            batch_user_ids = user_ids[i:i + batch_size]
            
            if show_progress:
                progress = (i + len(batch_user_ids)) / total_users * 100
                logger.info(f"Processing batch {i//batch_size + 1}: "
                           f"{progress:.1f}% complete")
            
            # Process each user in the batch
            with torch.no_grad():
                for user_id in batch_user_ids:
                    try:
                        # Retrieve candidates
                        candidates = self.retrieval_service.get_candidates(
                            user_id=user_id,
                            num_candidates=100
                        )
                        
                        # Rank candidates
                        ranked = self.ranking_service.rank(
                            user_id=user_id,
                            item_ids=candidates,
                            num_results=num_recommendations
                        )
                        
                        results[user_id] = ranked
                        
                    except Exception as e:
                        logger.error(f"Error processing user {user_id}: {e}")
                        results[user_id] = []
        
        elapsed = time.time() - start_time
        throughput = total_users / elapsed
        
        logger.info(f"Completed {total_users} users in {elapsed:.2f}s "
                   f"({throughput:.1f} users/sec)")
        
        return results
    
    def export_results(
        self,
        results: Dict[int, List[Dict]],
        output_path: str,
        format: str = "auto"
    ):
        """
        Export results to file.
        
        Args:
            results: Recommendation results
            output_path: Output file path
            format: Output format ('json', 'parquet', 'csv', or 'auto')
        """
        output_path = Path(output_path)
        
        # Auto-detect format from extension
        if format == "auto":
            format = output_path.suffix.lstrip('.')
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
        elif format == "parquet":
            import pandas as pd
            
            # Flatten results
            rows = []
            for user_id, recs in results.items():
                for rank, rec in enumerate(recs, 1):
                    rows.append({
                        'user_id': user_id,
                        'rank': rank,
                        'item_id': rec['item_id'],
                        'score': rec['score']
                    })
            
            df = pd.DataFrame(rows)
            df.to_parquet(output_path, index=False)
            
        elif format == "csv":
            import pandas as pd
            
            # Flatten results
            rows = []
            for user_id, recs in results.items():
                for rank, rec in enumerate(recs, 1):
                    rows.append({
                        'user_id': user_id,
                        'rank': rank,
                        'item_id': rec['item_id'],
                        'score': rec['score']
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results exported to {output_path}")


def load_user_ids_from_file(file_path: str) -> List[int]:
    """Load user IDs from CSV or text file."""
    import pandas as pd
    
    path = Path(file_path)
    
    if path.suffix == '.csv':
        df = pd.read_csv(path)
        # Assume first column or 'user_id' column
        if 'user_id' in df.columns:
            return df['user_id'].tolist()
        else:
            return df.iloc[:, 0].tolist()
    else:
        # Plain text file, one ID per line
        with open(path) as f:
            return [int(line.strip()) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Batch recommendation inference"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Input file with user IDs (CSV or text)"
    )
    input_group.add_argument(
        "--user-ids",
        type=str,
        help="Comma-separated list of user IDs"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (.json, .parquet, or .csv)"
    )
    parser.add_argument(
        "--num-recs",
        type=int,
        default=10,
        help="Number of recommendations per user (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Processing batch size (default: 64)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/checkpoints/two_tower/best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Load user IDs
    if args.input:
        user_ids = load_user_ids_from_file(args.input)
    else:
        user_ids = [int(x.strip()) for x in args.user_ids.split(',')]
    
    logger.info(f"Processing {len(user_ids)} users")
    
    # Initialize recommender
    recommender = BatchRecommender(
        model_path=args.model_path,
        device=args.device
    )
    
    # Generate recommendations
    results = recommender.recommend_batch(
        user_ids=user_ids,
        num_recommendations=args.num_recs,
        batch_size=args.batch_size
    )
    
    # Export results
    recommender.export_results(results, args.output)
    
    print(f"\n✅ Generated recommendations for {len(user_ids)} users")
    print(f"   Output saved to: {args.output}")


if __name__ == "__main__":
    main()
