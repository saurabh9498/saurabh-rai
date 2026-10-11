"""
Feature Transformations

GPU-accelerated feature transformations using NVTabular patterns.
Provides composable transformers for preprocessing recommendation data.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Try GPU imports, fall back to CPU
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    GPU_AVAILABLE = False
    logger.info("GPU libraries not available, using CPU fallback")


@dataclass
class ColumnSchema:
    """Schema for a feature column."""
    name: str
    dtype: str
    is_categorical: bool = False
    is_list: bool = False
    cardinality: Optional[int] = None
    tags: List[str] = None


class BaseTransformer(ABC):
    """Base class for feature transformers."""
    
    @abstractmethod
    def fit(self, df: Any) -> 'BaseTransformer':
        """Fit transformer on data."""
        pass
    
    @abstractmethod
    def transform(self, df: Any) -> Any:
        """Transform data."""
        pass
    
    def fit_transform(self, df: Any) -> Any:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class Categorify(BaseTransformer):
    """
    Convert categorical columns to integer indices.
    
    Similar to NVTabular's Categorify operator.
    """
    
    def __init__(
        self,
        columns: List[str],
        freq_threshold: int = 0,
        num_buckets: Optional[int] = None,
        na_value: str = "__NULL__",
    ):
        self.columns = columns
        self.freq_threshold = freq_threshold
        self.num_buckets = num_buckets
        self.na_value = na_value
        self.mappings: Dict[str, Dict] = {}
        self.cardinalities: Dict[str, int] = {}
    
    def fit(self, df: Any) -> 'Categorify':
        """Build vocabulary for each column."""
        for col in self.columns:
            if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
                series = df[col].to_pandas()
            else:
                series = df[col]
            
            # Count frequencies
            value_counts = series.value_counts()
            
            # Filter by frequency threshold
            if self.freq_threshold > 0:
                value_counts = value_counts[value_counts >= self.freq_threshold]
            
            # Build mapping (0 reserved for unknown/padding)
            unique_values = value_counts.index.tolist()
            
            if self.num_buckets and len(unique_values) > self.num_buckets:
                unique_values = unique_values[:self.num_buckets]
            
            self.mappings[col] = {
                val: idx + 1 for idx, val in enumerate(unique_values)
            }
            self.cardinalities[col] = len(unique_values) + 1  # +1 for unknown
        
        return self
    
    def transform(self, df: Any) -> Any:
        """Apply vocabulary mapping."""
        result = df.copy()
        
        for col in self.columns:
            mapping = self.mappings[col]
            
            if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
                # GPU path
                result[col] = df[col].map(mapping).fillna(0).astype('int32')
            else:
                # CPU path
                result[col] = df[col].map(mapping).fillna(0).astype('int32')
        
        return result


class Normalize(BaseTransformer):
    """
    Normalize numerical columns.
    
    Supports: standard (z-score), minmax, log1p
    """
    
    def __init__(
        self,
        columns: List[str],
        method: str = "standard",
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ):
        self.columns = columns
        self.method = method
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.stats: Dict[str, Dict[str, float]] = {}
    
    def fit(self, df: Any) -> 'Normalize':
        """Compute normalization statistics."""
        for col in self.columns:
            if GPU_AVAILABLE and hasattr(df[col], 'values_host'):
                values = df[col].values_host
            else:
                values = df[col].values
            
            if self.method == "standard":
                self.stats[col] = {
                    'mean': float(np.nanmean(values)),
                    'std': float(np.nanstd(values)) + 1e-8,
                }
            elif self.method == "minmax":
                self.stats[col] = {
                    'min': float(np.nanmin(values)),
                    'max': float(np.nanmax(values)) + 1e-8,
                }
            elif self.method == "log1p":
                self.stats[col] = {}  # No stats needed
        
        return self
    
    def transform(self, df: Any) -> Any:
        """Apply normalization."""
        result = df.copy()
        
        for col in self.columns:
            values = result[col]
            
            if self.method == "standard":
                stats = self.stats[col]
                values = (values - stats['mean']) / stats['std']
            elif self.method == "minmax":
                stats = self.stats[col]
                values = (values - stats['min']) / (stats['max'] - stats['min'])
            elif self.method == "log1p":
                if GPU_AVAILABLE and hasattr(values, 'log1p'):
                    values = values.clip(lower=0).log1p()
                else:
                    values = np.log1p(np.clip(values, 0, None))
            
            # Clip if specified
            if self.clip_min is not None or self.clip_max is not None:
                if GPU_AVAILABLE and hasattr(values, 'clip'):
                    values = values.clip(self.clip_min, self.clip_max)
                else:
                    values = np.clip(values, self.clip_min, self.clip_max)
            
            result[col] = values
        
        return result


class FillMissing(BaseTransformer):
    """Fill missing values in columns."""
    
    def __init__(
        self,
        columns: List[str],
        fill_value: Union[float, str] = 0,
        strategy: str = "constant",  # constant, mean, median, mode
    ):
        self.columns = columns
        self.fill_value = fill_value
        self.strategy = strategy
        self.fill_values: Dict[str, Any] = {}
    
    def fit(self, df: Any) -> 'FillMissing':
        """Compute fill values if using mean/median/mode."""
        for col in self.columns:
            if self.strategy == "constant":
                self.fill_values[col] = self.fill_value
            elif self.strategy == "mean":
                self.fill_values[col] = df[col].mean()
            elif self.strategy == "median":
                self.fill_values[col] = df[col].median()
            elif self.strategy == "mode":
                self.fill_values[col] = df[col].mode()[0]
        
        return self
    
    def transform(self, df: Any) -> Any:
        """Fill missing values."""
        result = df.copy()
        
        for col in self.columns:
            result[col] = result[col].fillna(self.fill_values[col])
        
        return result


class Bucketize(BaseTransformer):
    """Convert numerical columns to buckets."""
    
    def __init__(
        self,
        columns: List[str],
        boundaries: Optional[List[float]] = None,
        num_buckets: int = 10,
        method: str = "uniform",  # uniform, quantile
    ):
        self.columns = columns
        self.boundaries = boundaries
        self.num_buckets = num_buckets
        self.method = method
        self.bucket_boundaries: Dict[str, List[float]] = {}
    
    def fit(self, df: Any) -> 'Bucketize':
        """Compute bucket boundaries."""
        for col in self.columns:
            if self.boundaries:
                self.bucket_boundaries[col] = self.boundaries
            else:
                if GPU_AVAILABLE and hasattr(df[col], 'values_host'):
                    values = df[col].values_host
                else:
                    values = df[col].values
                
                if self.method == "uniform":
                    min_val = np.nanmin(values)
                    max_val = np.nanmax(values)
                    self.bucket_boundaries[col] = np.linspace(
                        min_val, max_val, self.num_buckets + 1
                    )[1:-1].tolist()
                else:  # quantile
                    percentiles = np.linspace(0, 100, self.num_buckets + 1)[1:-1]
                    self.bucket_boundaries[col] = np.nanpercentile(
                        values, percentiles
                    ).tolist()
        
        return self
    
    def transform(self, df: Any) -> Any:
        """Apply bucketization."""
        result = df.copy()
        
        for col in self.columns:
            boundaries = self.bucket_boundaries[col]
            
            if GPU_AVAILABLE and hasattr(df[col], 'values_host'):
                values = df[col].values_host
            else:
                values = df[col].values
            
            bucket_indices = np.digitize(values, boundaries)
            result[col] = bucket_indices
        
        return result


class TargetEncoding(BaseTransformer):
    """Target encoding for categorical features."""
    
    def __init__(
        self,
        columns: List[str],
        target_column: str,
        smoothing: float = 10.0,
    ):
        self.columns = columns
        self.target_column = target_column
        self.smoothing = smoothing
        self.encodings: Dict[str, Dict] = {}
        self.global_mean: float = 0.0
    
    def fit(self, df: Any) -> 'TargetEncoding':
        """Compute target encodings."""
        if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
            pdf = df.to_pandas()
        else:
            pdf = df
        
        self.global_mean = pdf[self.target_column].mean()
        
        for col in self.columns:
            grouped = pdf.groupby(col)[self.target_column].agg(['mean', 'count'])
            
            # Smoothed encoding
            smoothed = (
                grouped['count'] * grouped['mean'] + 
                self.smoothing * self.global_mean
            ) / (grouped['count'] + self.smoothing)
            
            self.encodings[col] = smoothed.to_dict()
        
        return self
    
    def transform(self, df: Any) -> Any:
        """Apply target encoding."""
        result = df.copy()
        
        for col in self.columns:
            encoding = self.encodings[col]
            new_col_name = f"{col}_target_enc"
            
            if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
                result[new_col_name] = df[col].map(encoding).fillna(self.global_mean)
            else:
                result[new_col_name] = df[col].map(encoding).fillna(self.global_mean)
        
        return result


class Pipeline(BaseTransformer):
    """Chain multiple transformers together."""
    
    def __init__(self, transformers: List[BaseTransformer]):
        self.transformers = transformers
    
    def fit(self, df: Any) -> 'Pipeline':
        """Fit all transformers."""
        current = df
        for transformer in self.transformers:
            transformer.fit(current)
            current = transformer.transform(current)
        return self
    
    def transform(self, df: Any) -> Any:
        """Apply all transformers."""
        current = df
        for transformer in self.transformers:
            current = transformer.transform(current)
        return current
    
    def add(self, transformer: BaseTransformer) -> 'Pipeline':
        """Add a transformer to the pipeline."""
        self.transformers.append(transformer)
        return self


def create_default_pipeline(
    categorical_columns: List[str],
    numerical_columns: List[str],
    target_column: Optional[str] = None,
) -> Pipeline:
    """Create a default preprocessing pipeline."""
    transformers = [
        FillMissing(numerical_columns, fill_value=0),
        Normalize(numerical_columns, method="standard"),
        Categorify(categorical_columns, freq_threshold=10),
    ]
    
    if target_column:
        transformers.append(
            TargetEncoding(categorical_columns, target_column)
        )
    
    return Pipeline(transformers)
