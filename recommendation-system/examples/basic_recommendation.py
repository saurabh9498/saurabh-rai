#!/usr/bin/env python3
"""
Basic Recommendation Example

This example demonstrates how to:
1. Load a trained recommendation model
2. Get recommendations for a single user
3. Filter and rank results

Usage:
    python examples/basic_recommendation.py --user-id 12345
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerModel
from src.serving.retrieval import RetrievalService
from src.serving.ranking import RankingService
from src.data.data_loader import DataLoader
from src.utils.config import load_config


def get_recommendations(
    user_id: int,
    num_recommendations: int = 10,
    model_path: str = "models/checkpoints/two_tower/best.pt",
    config_path: str = "configs/model_config.yaml"
) -> list:
    """
    Get personalized recommendations for a user.
    
    Args:
        user_id: The user ID to get recommendations for
        num_recommendations: Number of items to recommend
        model_path: Path to trained model checkpoint
        config_path: Path to model configuration
        
    Returns:
        List of recommended items with scores
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize model
    print(f"Loading model from {model_path}...")
    model = TwoTowerModel.load(model_path)
    model.eval()
    
    # Initialize services
    retrieval_service = RetrievalService(model, config)
    ranking_service = RankingService(model, config)
    
    # Step 1: Retrieve candidate items (fast, approximate)
    print(f"Retrieving candidates for user {user_id}...")
    candidates = retrieval_service.get_candidates(
        user_id=user_id,
        num_candidates=100  # Get more candidates for ranking
    )
    
    # Step 2: Rank candidates (slower, more accurate)
    print("Ranking candidates...")
    ranked_items = ranking_service.rank(
        user_id=user_id,
        item_ids=candidates,
        num_results=num_recommendations
    )
    
    return ranked_items


def main():
    parser = argparse.ArgumentParser(
        description="Get recommendations for a user"
    )
    parser.add_argument(
        "--user-id",
        type=int,
        required=True,
        help="User ID to get recommendations for"
    )
    parser.add_argument(
        "--num-recs",
        type=int,
        default=10,
        help="Number of recommendations (default: 10)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/checkpoints/two_tower/best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (JSON)"
    )
    
    args = parser.parse_args()
    
    # Get recommendations
    recommendations = get_recommendations(
        user_id=args.user_id,
        num_recommendations=args.num_recs,
        model_path=args.model_path
    )
    
    # Display results
    print(f"\n{'='*50}")
    print(f"Top {args.num_recs} Recommendations for User {args.user_id}")
    print(f"{'='*50}\n")
    
    for i, item in enumerate(recommendations, 1):
        print(f"{i:2}. Item ID: {item['item_id']:8} | Score: {item['score']:.4f}")
        if 'title' in item:
            print(f"    Title: {item['title']}")
        if 'category' in item:
            print(f"    Category: {item['category']}")
        print()
    
    # Save to file if requested
    if args.output:
        output_data = {
            "user_id": args.user_id,
            "recommendations": recommendations
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
