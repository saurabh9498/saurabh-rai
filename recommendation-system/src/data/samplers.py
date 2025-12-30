"""
Negative Sampling Strategies

Provides various negative sampling methods for recommendation training:
- Uniform random sampling
- Popularity-based sampling
- Hard negative mining
- In-batch negatives
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseSampler:
    """Base class for negative samplers."""
    
    def __init__(self, num_items: int, seed: int = 42):
        self.num_items = num_items
        self.rng = np.random.RandomState(seed)
    
    def sample(
        self,
        positive_items: List[int],
        num_negatives: int,
    ) -> np.ndarray:
        """
        Sample negative items.
        
        Args:
            positive_items: List of positive item indices to exclude
            num_negatives: Number of negatives to sample
            
        Returns:
            Array of negative item indices
        """
        raise NotImplementedError


class UniformSampler(BaseSampler):
    """Uniform random negative sampling."""
    
    def sample(
        self,
        positive_items: List[int],
        num_negatives: int,
    ) -> np.ndarray:
        """Sample uniformly at random, excluding positives."""
        positive_set = set(positive_items)
        negatives = []
        
        while len(negatives) < num_negatives:
            candidate = self.rng.randint(0, self.num_items)
            if candidate not in positive_set:
                negatives.append(candidate)
        
        return np.array(negatives)
    
    def batch_sample(
        self,
        positive_items_batch: List[List[int]],
        num_negatives: int,
    ) -> np.ndarray:
        """Sample negatives for a batch of positive sets."""
        batch_negatives = []
        
        for positive_items in positive_items_batch:
            negatives = self.sample(positive_items, num_negatives)
            batch_negatives.append(negatives)
        
        return np.stack(batch_negatives)


class PopularitySampler(BaseSampler):
    """
    Popularity-biased negative sampling.
    
    More popular items are more likely to be sampled as negatives,
    making the model better at distinguishing from popular baselines.
    """
    
    def __init__(
        self,
        num_items: int,
        item_counts: np.ndarray,
        smoothing: float = 0.75,
        seed: int = 42,
    ):
        super().__init__(num_items, seed)
        
        # Smoothed popularity distribution
        counts = np.array(item_counts) + 1  # Add-one smoothing
        counts = counts ** smoothing
        self.probs = counts / counts.sum()
    
    def sample(
        self,
        positive_items: List[int],
        num_negatives: int,
    ) -> np.ndarray:
        """Sample based on item popularity."""
        positive_set = set(positive_items)
        
        # Adjust probabilities to exclude positives
        adjusted_probs = self.probs.copy()
        for pos in positive_items:
            if pos < len(adjusted_probs):
                adjusted_probs[pos] = 0
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        negatives = self.rng.choice(
            self.num_items,
            size=num_negatives,
            replace=True,  # Allow replacement for efficiency
            p=adjusted_probs,
        )
        
        return negatives


class HardNegativeSampler(BaseSampler):
    """
    Hard negative mining sampler.
    
    Samples negatives that are similar to positives (harder to distinguish).
    Requires item embeddings or similarity matrix.
    """
    
    def __init__(
        self,
        num_items: int,
        item_embeddings: np.ndarray,
        temperature: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(num_items, seed)
        self.embeddings = item_embeddings
        self.temperature = temperature
        
        # Normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
    
    def sample(
        self,
        positive_items: List[int],
        num_negatives: int,
    ) -> np.ndarray:
        """Sample hard negatives based on similarity to positives."""
        if not positive_items:
            # Fall back to uniform
            return self.rng.randint(0, self.num_items, num_negatives)
        
        # Get positive embeddings
        pos_embeds = self.embeddings[positive_items]
        avg_pos_embed = pos_embeds.mean(axis=0)
        
        # Compute similarities to all items
        similarities = self.embeddings @ avg_pos_embed
        
        # Convert to sampling probabilities
        similarities = similarities / self.temperature
        exp_sims = np.exp(similarities - similarities.max())
        
        # Zero out positives
        for pos in positive_items:
            exp_sims[pos] = 0
        
        probs = exp_sims / exp_sims.sum()
        
        negatives = self.rng.choice(
            self.num_items,
            size=num_negatives,
            replace=True,
            p=probs,
        )
        
        return negatives


class MixedSampler(BaseSampler):
    """
    Mixed negative sampling strategy.
    
    Combines easy (uniform) and hard negatives.
    """
    
    def __init__(
        self,
        num_items: int,
        item_embeddings: Optional[np.ndarray] = None,
        hard_negative_ratio: float = 0.5,
        seed: int = 42,
    ):
        super().__init__(num_items, seed)
        
        self.uniform_sampler = UniformSampler(num_items, seed)
        
        self.hard_sampler = None
        if item_embeddings is not None:
            self.hard_sampler = HardNegativeSampler(
                num_items, item_embeddings, seed=seed
            )
        
        self.hard_ratio = hard_negative_ratio
    
    def sample(
        self,
        positive_items: List[int],
        num_negatives: int,
    ) -> np.ndarray:
        """Sample mix of easy and hard negatives."""
        if self.hard_sampler is None:
            return self.uniform_sampler.sample(positive_items, num_negatives)
        
        num_hard = int(num_negatives * self.hard_ratio)
        num_easy = num_negatives - num_hard
        
        hard_negatives = self.hard_sampler.sample(positive_items, num_hard)
        easy_negatives = self.uniform_sampler.sample(positive_items, num_easy)
        
        return np.concatenate([hard_negatives, easy_negatives])


class InBatchNegativeSampler:
    """
    In-batch negative sampling for contrastive learning.
    
    Uses other items in the batch as negatives.
    Very efficient for two-tower models.
    """
    
    def __init__(self, exclude_diagonal: bool = True):
        self.exclude_diagonal = exclude_diagonal
    
    def get_negative_mask(self, batch_size: int) -> np.ndarray:
        """
        Get mask for valid negatives in batch.
        
        Returns:
            (batch_size, batch_size) boolean mask
        """
        mask = np.ones((batch_size, batch_size), dtype=bool)
        
        if self.exclude_diagonal:
            np.fill_diagonal(mask, False)
        
        return mask
    
    def compute_logits(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute logits with in-batch negatives.
        
        Args:
            user_embeddings: (batch, dim) user representations
            item_embeddings: (batch, dim) item representations
            temperature: Softmax temperature
            
        Returns:
            logits: (batch, batch) similarity matrix
            labels: (batch,) indices of positive items (diagonal)
        """
        # Compute all pairwise similarities
        logits = np.dot(user_embeddings, item_embeddings.T) / temperature
        
        # Labels are the diagonal (each user's positive is their own item)
        labels = np.arange(len(user_embeddings))
        
        return logits, labels


class NegativeSamplingDataset:
    """
    Dataset wrapper that adds negative samples.
    """
    
    def __init__(
        self,
        positive_pairs: List[Tuple[int, int]],
        num_items: int,
        sampler: BaseSampler,
        num_negatives: int = 4,
    ):
        self.positive_pairs = positive_pairs
        self.num_items = num_items
        self.sampler = sampler
        self.num_negatives = num_negatives
        
        # Build user -> items mapping for efficient sampling
        self.user_items: Dict[int, List[int]] = {}
        for user, item in positive_pairs:
            if user not in self.user_items:
                self.user_items[user] = []
            self.user_items[user].append(item)
    
    def __len__(self) -> int:
        return len(self.positive_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get training example with negatives.
        
        Returns:
            Dict with user, positive_item, negative_items, labels
        """
        user, positive_item = self.positive_pairs[idx]
        
        # Sample negatives
        user_positives = self.user_items.get(user, [positive_item])
        negative_items = self.sampler.sample(user_positives, self.num_negatives)
        
        return {
            'user': np.array(user),
            'positive_item': np.array(positive_item),
            'negative_items': negative_items,
            'labels': np.array([1] + [0] * self.num_negatives),
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, np.ndarray]:
        """Collate batch of examples."""
        return {
            'users': np.stack([b['user'] for b in batch]),
            'positive_items': np.stack([b['positive_item'] for b in batch]),
            'negative_items': np.stack([b['negative_items'] for b in batch]),
            'labels': np.stack([b['labels'] for b in batch]),
        }


def create_sampler(
    strategy: str,
    num_items: int,
    item_counts: Optional[np.ndarray] = None,
    item_embeddings: Optional[np.ndarray] = None,
    **kwargs,
) -> BaseSampler:
    """
    Factory function to create a sampler.
    
    Args:
        strategy: Sampling strategy (uniform, popularity, hard, mixed)
        num_items: Total number of items
        item_counts: Item frequency counts (for popularity sampling)
        item_embeddings: Item embeddings (for hard negative sampling)
        **kwargs: Additional sampler-specific arguments
        
    Returns:
        BaseSampler instance
    """
    if strategy == "uniform":
        return UniformSampler(num_items, **kwargs)
    elif strategy == "popularity":
        if item_counts is None:
            raise ValueError("item_counts required for popularity sampling")
        return PopularitySampler(num_items, item_counts, **kwargs)
    elif strategy == "hard":
        if item_embeddings is None:
            raise ValueError("item_embeddings required for hard negative sampling")
        return HardNegativeSampler(num_items, item_embeddings, **kwargs)
    elif strategy == "mixed":
        return MixedSampler(num_items, item_embeddings, **kwargs)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
