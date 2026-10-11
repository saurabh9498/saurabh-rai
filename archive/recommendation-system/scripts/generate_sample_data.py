#!/usr/bin/env python3
"""
Generate synthetic sample data for the Recommendation System.

This script creates realistic synthetic datasets for development and testing,
including users, items, and user-item interactions.

Usage:
    python scripts/generate_sample_data.py --users 10000 --items 5000 --interactions 500000
    python scripts/generate_sample_data.py --output data/sample/ --seed 42
"""

import argparse
import hashlib
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Configuration
COUNTRIES = ["US", "GB", "DE", "FR", "JP", "CA", "AU", "BR", "IN", "MX"]
DEVICE_TYPES = ["mobile", "desktop", "tablet", "smart_tv"]
BROWSERS = ["chrome", "safari", "firefox", "edge", "app"]
EVENT_TYPES = ["view", "click", "add_to_cart", "purchase"]
EVENT_WEIGHTS = [0.70, 0.20, 0.07, 0.03]  # Probability distribution

CATEGORY_L1 = ["Electronics", "Fashion", "Home", "Sports", "Beauty", "Books", "Toys"]
CATEGORY_L2 = {
    "Electronics": ["Phones", "Laptops", "Audio", "Cameras", "Gaming"],
    "Fashion": ["Men", "Women", "Kids", "Accessories", "Shoes"],
    "Home": ["Furniture", "Kitchen", "Decor", "Garden", "Bedding"],
    "Sports": ["Fitness", "Outdoor", "Team Sports", "Water Sports", "Cycling"],
    "Beauty": ["Skincare", "Makeup", "Haircare", "Fragrance", "Personal Care"],
    "Books": ["Fiction", "Non-Fiction", "Technical", "Children", "Comics"],
    "Toys": ["Action Figures", "Board Games", "Educational", "Outdoor", "Dolls"],
}

BRANDS = [
    "TechPro", "StyleMax", "HomeEssentials", "SportX", "BeautyGlow",
    "ReadMore", "PlayTime", "ValueBrand", "PremiumChoice", "EcoFriendly",
    "InnovateTech", "FashionForward", "ComfortLiving", "ActiveLife", "NaturalBeauty"
]


class DataGenerator:
    """Generates synthetic recommendation system data."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 12, 28)

    def generate_user_id(self, idx: int) -> str:
        """Generate deterministic user ID."""
        return f"user_{idx:06d}"

    def generate_item_id(self, idx: int) -> str:
        """Generate deterministic item ID."""
        return f"item_{idx:05d}"

    def generate_users(self, num_users: int) -> pd.DataFrame:
        """Generate user profiles dataset."""
        print(f"  Generating {num_users:,} users...")

        users = []
        for i in range(num_users):
            # Determine user segment (affects behavior patterns)
            segment = np.random.choice(
                ["power", "active", "casual", "low"],
                p=[0.05, 0.25, 0.40, 0.30]
            )

            # Signup date weighted toward earlier dates
            days_ago = int(np.random.exponential(180))
            days_ago = min(days_ago, (self.end_date - self.start_date).days)
            signup_date = self.end_date - timedelta(days=days_ago)

            # Activity level correlates with segment
            activity_map = {"power": 5, "active": 4, "casual": 2, "low": 1}
            activity_level = activity_map[segment] + random.randint(-1, 1)
            activity_level = max(1, min(5, activity_level))

            # LTV correlates with activity
            base_ltv = {"power": 500, "active": 200, "casual": 50, "low": 10}
            lifetime_value = base_ltv[segment] * np.random.lognormal(0, 0.5)

            # Category preferences (2-4 categories)
            num_prefs = random.randint(2, 4)
            preferred_categories = random.sample(CATEGORY_L1, num_prefs)

            user = {
                "user_id": self.generate_user_id(i),
                "age_bucket": random.randint(0, 5),
                "gender": random.randint(0, 2),
                "country": random.choice(COUNTRIES),
                "signup_date": signup_date,
                "lifetime_value": round(lifetime_value, 2),
                "activity_level": activity_level,
                "preferred_categories": preferred_categories,
                "device_type": random.choice(DEVICE_TYPES),
                "subscription_tier": np.random.choice([0, 1, 2], p=[0.7, 0.25, 0.05]),
                "_segment": segment,  # Internal use for interaction generation
            }
            users.append(user)

        df = pd.DataFrame(users)
        print(f"    ✓ Users generated")
        return df

    def generate_items(self, num_items: int) -> pd.DataFrame:
        """Generate item catalog dataset."""
        print(f"  Generating {num_items:,} items...")

        items = []
        for i in range(num_items):
            category_l1 = random.choice(CATEGORY_L1)
            category_l2 = random.choice(CATEGORY_L2[category_l1])
            category_l3 = f"{category_l2}_{random.randint(1, 10)}"

            # Price distribution (log-normal)
            price = np.random.lognormal(3.5, 1.0)
            price = max(5.0, min(2000.0, price))

            # Rating (beta distribution, skewed toward higher ratings)
            avg_rating = np.random.beta(5, 2) * 4 + 1  # Range 1-5

            # Review count (power law)
            review_count = int(np.random.pareto(1.5) * 10)
            review_count = min(review_count, 10000)

            # Created date
            days_ago = int(np.random.exponential(90))
            days_ago = min(days_ago, 365)
            created_at = self.end_date - timedelta(days=days_ago)

            # Popularity correlates with reviews and rating
            popularity_score = (review_count ** 0.3) * (avg_rating / 5) * random.uniform(0.8, 1.2)

            # Content embedding (128-dim, normalized)
            embedding = np.random.randn(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            item = {
                "item_id": self.generate_item_id(i),
                "category_l1": category_l1,
                "category_l2": category_l2,
                "category_l3": category_l3,
                "brand": random.choice(BRANDS),
                "price": round(price, 2),
                "avg_rating": round(avg_rating, 2),
                "review_count": review_count,
                "created_at": created_at,
                "popularity_score": round(popularity_score, 4),
                "embedding": embedding.tolist(),
            }
            items.append(item)

        df = pd.DataFrame(items)
        print(f"    ✓ Items generated")
        return df

    def generate_interactions(
        self,
        num_interactions: int,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate user-item interaction events."""
        print(f"  Generating {num_interactions:,} interactions...")

        # Pre-compute user segments for faster lookup
        user_segments = dict(zip(users_df["user_id"], users_df["_segment"]))
        user_prefs = dict(zip(users_df["user_id"], users_df["preferred_categories"]))
        item_categories = dict(zip(items_df["item_id"], items_df["category_l1"]))

        # Interaction counts per segment
        interactions_per_segment = {
            "power": 100,
            "active": 50,
            "casual": 15,
            "low": 5,
        }

        interactions = []
        user_ids = users_df["user_id"].tolist()
        item_ids = items_df["item_id"].tolist()

        # Weight items by popularity for more realistic distribution
        item_weights = (items_df["popularity_score"] ** 0.5).values
        item_weights = item_weights / item_weights.sum()

        generated = 0
        session_counter = 0

        while generated < num_interactions:
            # Select user (weighted by segment activity)
            user_id = random.choice(user_ids)
            segment = user_segments[user_id]
            preferred_cats = user_prefs[user_id]

            # Number of interactions in this session
            session_size = random.randint(1, min(20, interactions_per_segment[segment]))
            session_id = f"session_{session_counter:08d}"
            session_counter += 1

            # Session timestamp
            days_ago = random.randint(0, 365)
            session_start = self.end_date - timedelta(days=days_ago)
            session_start = session_start.replace(
                hour=random.randint(6, 23),
                minute=random.randint(0, 59),
            )

            # Device for this session
            device = random.choice(DEVICE_TYPES)
            browser = random.choice(BROWSERS)

            # Generate session interactions
            for pos in range(session_size):
                if generated >= num_interactions:
                    break

                # Item selection (biased toward user preferences)
                if random.random() < 0.6:  # 60% chance to pick from preferred category
                    preferred_items = [
                        iid for iid in item_ids
                        if item_categories[iid] in preferred_cats
                    ]
                    if preferred_items:
                        item_id = random.choice(preferred_items)
                    else:
                        item_id = np.random.choice(item_ids, p=item_weights)
                else:
                    item_id = np.random.choice(item_ids, p=item_weights)

                # Event type (sequential funnel)
                if pos == 0:
                    event_type = "view"
                else:
                    event_type = np.random.choice(EVENT_TYPES, p=EVENT_WEIGHTS)

                # Timestamp (incremental within session)
                timestamp = session_start + timedelta(seconds=pos * random.randint(10, 120))

                # Label (1 for purchase/add_to_cart, 0 otherwise)
                label = 1 if event_type in ["purchase", "add_to_cart"] else 0

                interaction = {
                    "user_id": user_id,
                    "item_id": item_id,
                    "event_type": event_type,
                    "timestamp": timestamp,
                    "session_id": session_id,
                    "position": pos,
                    "context_device": device,
                    "context_browser": browser,
                    "context_page": random.choice(["home", "search", "category", "pdp", "cart"]),
                    "label": label,
                }
                interactions.append(interaction)
                generated += 1

            # Progress update
            if generated % 100000 == 0:
                print(f"    ... {generated:,} / {num_interactions:,}")

        df = pd.DataFrame(interactions)
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"    ✓ Interactions generated")
        return df

    def save_datasets(
        self,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """Save datasets to parquet files."""
        print(f"  Saving to {output_dir}/...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Remove internal columns before saving
        users_clean = users_df.drop(columns=["_segment"])

        # Save as parquet
        users_clean.to_parquet(output_dir / "users.parquet", index=False)
        items_df.to_parquet(output_dir / "items.parquet", index=False)
        interactions_df.to_parquet(output_dir / "interactions.parquet", index=False)

        print(f"    ✓ users.parquet ({len(users_df):,} rows)")
        print(f"    ✓ items.parquet ({len(items_df):,} rows)")
        print(f"    ✓ interactions.parquet ({len(interactions_df):,} rows)")


def print_statistics(
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("📊 Dataset Statistics")
    print("=" * 60)

    print("\n👥 Users:")
    print(f"  Total users: {len(users_df):,}")
    print(f"  Countries: {users_df['country'].nunique()}")
    print(f"  Avg LTV: ${users_df['lifetime_value'].mean():.2f}")

    print("\n📦 Items:")
    print(f"  Total items: {len(items_df):,}")
    print(f"  Categories: {items_df['category_l1'].nunique()}")
    print(f"  Brands: {items_df['brand'].nunique()}")
    print(f"  Price range: ${items_df['price'].min():.2f} - ${items_df['price'].max():.2f}")

    print("\n🔄 Interactions:")
    print(f"  Total events: {len(interactions_df):,}")
    print(f"  Unique users: {interactions_df['user_id'].nunique():,}")
    print(f"  Unique items: {interactions_df['item_id'].nunique():,}")
    print(f"  Sessions: {interactions_df['session_id'].nunique():,}")

    print("\n  Event distribution:")
    event_counts = interactions_df["event_type"].value_counts()
    for event, count in event_counts.items():
        pct = count / len(interactions_df) * 100
        print(f"    {event}: {count:,} ({pct:.1f}%)")

    print("\n  Label distribution:")
    label_counts = interactions_df["label"].value_counts()
    pos_rate = label_counts.get(1, 0) / len(interactions_df) * 100
    print(f"    Positive: {label_counts.get(1, 0):,} ({pos_rate:.1f}%)")
    print(f"    Negative: {label_counts.get(0, 0):,} ({100 - pos_rate:.1f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic recommendation system data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_sample_data.py
  python generate_sample_data.py --users 100000 --items 50000 --interactions 5000000
  python generate_sample_data.py --output data/large/ --seed 123
        """,
    )
    parser.add_argument(
        "--users", type=int, default=10000, help="Number of users (default: 10000)"
    )
    parser.add_argument(
        "--items", type=int, default=5000, help="Number of items (default: 5000)"
    )
    parser.add_argument(
        "--interactions",
        type=int,
        default=500000,
        help="Number of interactions (default: 500000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample"),
        help="Output directory (default: data/sample)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🎯 Recommendation System - Sample Data Generator")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Users: {args.users:,}")
    print(f"  Items: {args.items:,}")
    print(f"  Interactions: {args.interactions:,}")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print()

    # Generate data
    generator = DataGenerator(seed=args.seed)

    users_df = generator.generate_users(args.users)
    items_df = generator.generate_items(args.items)
    interactions_df = generator.generate_interactions(
        args.interactions, users_df, items_df
    )

    # Save datasets
    generator.save_datasets(users_df, items_df, interactions_df, args.output)

    # Print statistics
    print_statistics(users_df, items_df, interactions_df)

    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
