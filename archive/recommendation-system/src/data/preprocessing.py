"""
Data Preprocessing

Cleans and prepares recommendation data:
- Filtering (min interactions, valid users/items)
- Deduplication
- Timestamp processing
- Label generation
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# GPU/CPU compatibility
try:
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    cudf = pd
    GPU_AVAILABLE = False


class DataCleaner:
    """Cleans raw interaction data."""
    
    def __init__(
        self,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        max_sequence_length: int = 100,
    ):
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.max_sequence_length = max_sequence_length
    
    def clean(self, df: Any) -> Any:
        """
        Apply all cleaning steps.
        
        Args:
            df: Raw interaction DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        original_len = len(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Filter by interaction counts
        df = self.filter_by_counts(df)
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        logger.info(
            f"Cleaned data: {original_len:,} -> {len(df):,} "
            f"({len(df)/original_len:.1%} retained)"
        )
        
        return df
    
    def remove_duplicates(
        self,
        df: Any,
        subset: Optional[List[str]] = None,
        keep: str = "last",
    ) -> Any:
        """Remove duplicate interactions."""
        subset = subset or ['user_id', 'item_id', 'timestamp']
        subset = [c for c in subset if c in df.columns]
        
        original_len = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        
        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed:,} duplicate rows")
        
        return df
    
    def filter_by_counts(self, df: Any) -> Any:
        """Filter users and items with insufficient interactions."""
        # Iteratively filter until stable
        prev_len = 0
        iteration = 0
        max_iterations = 10
        
        while len(df) != prev_len and iteration < max_iterations:
            prev_len = len(df)
            iteration += 1
            
            # Filter users
            if self.min_user_interactions > 1:
                user_counts = df.groupby('user_id').size()
                valid_users = user_counts[user_counts >= self.min_user_interactions].index
                df = df[df['user_id'].isin(valid_users)]
            
            # Filter items
            if self.min_item_interactions > 1:
                item_counts = df.groupby('item_id').size()
                valid_items = item_counts[item_counts >= self.min_item_interactions].index
                df = df[df['item_id'].isin(valid_items)]
        
        return df
    
    def truncate_sequences(
        self,
        df: Any,
        user_col: str = 'user_id',
        timestamp_col: str = 'timestamp',
    ) -> Any:
        """Truncate user sequences to max length."""
        if timestamp_col not in df.columns:
            return df
        
        # Sort by user and timestamp
        df = df.sort_values([user_col, timestamp_col])
        
        # Keep only last N interactions per user
        df = df.groupby(user_col).tail(self.max_sequence_length)
        
        return df


class LabelGenerator:
    """Generates training labels from interactions."""
    
    def __init__(
        self,
        positive_events: List[str] = None,
        negative_sampling_ratio: float = 4.0,
    ):
        self.positive_events = positive_events or ['click', 'purchase', 'add_to_cart']
        self.negative_sampling_ratio = negative_sampling_ratio
    
    def generate_binary_labels(
        self,
        df: Any,
        event_col: str = 'event_type',
    ) -> Any:
        """Generate binary labels (1 for positive, 0 for negative)."""
        if event_col in df.columns:
            df['label'] = df[event_col].isin(self.positive_events).astype(int)
        else:
            # Assume all interactions are positive
            df['label'] = 1
        
        return df
    
    def generate_implicit_labels(
        self,
        df: Any,
        event_weights: Optional[Dict[str, float]] = None,
        event_col: str = 'event_type',
    ) -> Any:
        """Generate weighted implicit labels."""
        default_weights = {
            'view': 1.0,
            'click': 2.0,
            'add_to_cart': 3.0,
            'purchase': 5.0,
        }
        weights = event_weights or default_weights
        
        if event_col in df.columns:
            df['label'] = df[event_col].map(weights).fillna(1.0)
        else:
            df['label'] = 1.0
        
        return df
    
    def create_negative_samples(
        self,
        df: Any,
        all_items: List[str],
        user_col: str = 'user_id',
        item_col: str = 'item_id',
    ) -> Any:
        """Create negative samples for training."""
        # Get positive items per user
        user_items = df.groupby(user_col)[item_col].apply(set).to_dict()
        
        negative_samples = []
        all_items_set = set(all_items)
        
        for user_id, positive_items in user_items.items():
            # Available negatives
            negative_pool = list(all_items_set - positive_items)
            
            if not negative_pool:
                continue
            
            # Sample negatives
            num_negatives = int(len(positive_items) * self.negative_sampling_ratio)
            num_negatives = min(num_negatives, len(negative_pool))
            
            sampled_negatives = np.random.choice(
                negative_pool,
                size=num_negatives,
                replace=False,
            )
            
            for item_id in sampled_negatives:
                negative_samples.append({
                    user_col: user_id,
                    item_col: item_id,
                    'label': 0,
                })
        
        # Create negative DataFrame
        if GPU_AVAILABLE:
            neg_df = cudf.DataFrame(negative_samples)
        else:
            neg_df = pd.DataFrame(negative_samples)
        
        # Combine with positives
        df['label'] = 1
        combined = pd.concat([df, neg_df], ignore_index=True)
        
        logger.info(
            f"Created {len(neg_df):,} negative samples "
            f"(ratio: {len(neg_df)/len(df):.1f}:1)"
        )
        
        return combined


class SequenceBuilder:
    """Builds sequences for sequential recommendation."""
    
    def __init__(
        self,
        max_length: int = 50,
        padding_value: int = 0,
    ):
        self.max_length = max_length
        self.padding_value = padding_value
    
    def build_sequences(
        self,
        df: Any,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        timestamp_col: str = 'timestamp',
    ) -> Dict[str, np.ndarray]:
        """
        Build padded sequences for each user.
        
        Returns:
            Dict with 'sequences' and 'lengths' arrays
        """
        # Sort by user and timestamp
        if timestamp_col in df.columns:
            df = df.sort_values([user_col, timestamp_col])
        
        # Group by user
        user_sequences = df.groupby(user_col)[item_col].apply(list).to_dict()
        
        # Build padded sequences
        sequences = []
        lengths = []
        user_ids = []
        
        for user_id, items in user_sequences.items():
            # Truncate to max length
            items = items[-self.max_length:]
            length = len(items)
            
            # Pad
            padded = [self.padding_value] * (self.max_length - length) + items
            
            sequences.append(padded)
            lengths.append(length)
            user_ids.append(user_id)
        
        return {
            'user_ids': np.array(user_ids),
            'sequences': np.array(sequences),
            'lengths': np.array(lengths),
        }
    
    def create_training_examples(
        self,
        sequences: Dict[str, np.ndarray],
        prediction_length: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create training examples from sequences.
        
        For each sequence, creates (input_seq, target) pairs
        using sliding window.
        
        Returns:
            input_sequences, targets, lengths
        """
        all_inputs = []
        all_targets = []
        all_lengths = []
        
        for seq, length in zip(sequences['sequences'], sequences['lengths']):
            if length <= prediction_length:
                continue
            
            # Create sliding window examples
            for i in range(prediction_length, length):
                input_seq = seq[self.max_length - i:self.max_length]
                target = seq[self.max_length - i + prediction_length - 1]
                
                all_inputs.append(input_seq)
                all_targets.append(target)
                all_lengths.append(i)
        
        return (
            np.array(all_inputs),
            np.array(all_targets),
            np.array(all_lengths),
        )


class VocabularyBuilder:
    """Builds vocabularies for categorical features."""
    
    def __init__(
        self,
        min_frequency: int = 1,
        max_size: Optional[int] = None,
        unknown_token: str = "<UNK>",
        padding_token: str = "<PAD>",
    ):
        self.min_frequency = min_frequency
        self.max_size = max_size
        self.unknown_token = unknown_token
        self.padding_token = padding_token
        
        self.vocabularies: Dict[str, Dict[str, int]] = {}
    
    def fit(
        self,
        df: Any,
        columns: List[str],
    ) -> 'VocabularyBuilder':
        """Build vocabularies for specified columns."""
        for col in columns:
            if col not in df.columns:
                continue
            
            # Count frequencies
            if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
                value_counts = df[col].to_pandas().value_counts()
            else:
                value_counts = df[col].value_counts()
            
            # Filter by frequency
            if self.min_frequency > 1:
                value_counts = value_counts[value_counts >= self.min_frequency]
            
            # Limit size
            if self.max_size:
                value_counts = value_counts.head(self.max_size - 2)  # Reserve for special tokens
            
            # Build vocabulary
            vocab = {self.padding_token: 0, self.unknown_token: 1}
            for idx, value in enumerate(value_counts.index):
                vocab[value] = idx + 2
            
            self.vocabularies[col] = vocab
            logger.info(f"Built vocabulary for {col}: {len(vocab)} tokens")
        
        return self
    
    def transform(
        self,
        df: Any,
        columns: Optional[List[str]] = None,
    ) -> Any:
        """Apply vocabularies to encode columns."""
        columns = columns or list(self.vocabularies.keys())
        
        for col in columns:
            if col not in self.vocabularies or col not in df.columns:
                continue
            
            vocab = self.vocabularies[col]
            default = vocab.get(self.unknown_token, 1)
            
            df[col] = df[col].map(vocab).fillna(default).astype(int)
        
        return df
    
    def get_cardinality(self, column: str) -> int:
        """Get vocabulary size for a column."""
        return len(self.vocabularies.get(column, {}))
