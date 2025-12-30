"""
Data Loading Utilities

GPU-accelerated data loading using cuDF for large-scale recommendation data.
Supports Parquet, CSV, and streaming data sources.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Union
from dataclasses import dataclass
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
    cudf = pd  # Alias for compatibility
    logger.info("cuDF not available, using pandas")


@dataclass
class DataConfig:
    """Configuration for data loading."""
    path: str
    format: str = "parquet"  # parquet, csv, json
    columns: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    partition_cols: Optional[List[str]] = None
    chunk_size: int = 100000


class DataLoader:
    """GPU-accelerated data loader."""
    
    def __init__(
        self,
        use_gpu: bool = True,
        memory_limit: Optional[int] = None,
    ):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.memory_limit = memory_limit
        
        if self.use_gpu:
            logger.info("Using GPU-accelerated data loading (cuDF)")
        else:
            logger.info("Using CPU data loading (pandas)")
    
    def load(
        self,
        path: Union[str, Path],
        format: str = "parquet",
        columns: Optional[List[str]] = None,
        filters: Optional[List] = None,
        **kwargs,
    ) -> Any:
        """
        Load data from file or directory.
        
        Args:
            path: File or directory path
            format: Data format (parquet, csv, json)
            columns: Columns to load (None for all)
            filters: Row filters (Parquet only)
            **kwargs: Additional format-specific arguments
            
        Returns:
            DataFrame (cudf or pandas)
        """
        path = Path(path)
        
        if format == "parquet":
            return self._load_parquet(path, columns, filters, **kwargs)
        elif format == "csv":
            return self._load_csv(path, columns, **kwargs)
        elif format == "json":
            return self._load_json(path, columns, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _load_parquet(
        self,
        path: Path,
        columns: Optional[List[str]] = None,
        filters: Optional[List] = None,
        **kwargs,
    ) -> Any:
        """Load Parquet file(s)."""
        if self.use_gpu:
            df = cudf.read_parquet(
                str(path),
                columns=columns,
                filters=filters,
                **kwargs,
            )
        else:
            import pyarrow.parquet as pq
            
            if path.is_dir():
                # Load partitioned dataset
                table = pq.read_table(
                    str(path),
                    columns=columns,
                    filters=filters,
                )
                df = table.to_pandas()
            else:
                df = pd.read_parquet(
                    str(path),
                    columns=columns,
                    filters=filters,
                    **kwargs,
                )
        
        logger.info(f"Loaded {len(df):,} rows from {path}")
        return df
    
    def _load_csv(
        self,
        path: Path,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """Load CSV file(s)."""
        if self.use_gpu:
            df = cudf.read_csv(str(path), usecols=columns, **kwargs)
        else:
            df = pd.read_csv(str(path), usecols=columns, **kwargs)
        
        logger.info(f"Loaded {len(df):,} rows from {path}")
        return df
    
    def _load_json(
        self,
        path: Path,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """Load JSON file(s)."""
        if self.use_gpu:
            df = cudf.read_json(str(path), **kwargs)
        else:
            df = pd.read_json(str(path), **kwargs)
        
        if columns:
            df = df[columns]
        
        logger.info(f"Loaded {len(df):,} rows from {path}")
        return df
    
    def load_chunked(
        self,
        path: Union[str, Path],
        chunk_size: int = 100000,
        format: str = "parquet",
        **kwargs,
    ) -> Iterator[Any]:
        """
        Load data in chunks for memory efficiency.
        
        Yields:
            DataFrame chunks
        """
        path = Path(path)
        
        if format == "csv":
            if self.use_gpu:
                # cuDF doesn't support chunked reading natively
                for chunk in pd.read_csv(str(path), chunksize=chunk_size, **kwargs):
                    yield cudf.from_pandas(chunk)
            else:
                for chunk in pd.read_csv(str(path), chunksize=chunk_size, **kwargs):
                    yield chunk
        elif format == "parquet":
            # Read parquet in row groups
            import pyarrow.parquet as pq
            
            parquet_file = pq.ParquetFile(str(path))
            
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                if self.use_gpu:
                    yield cudf.from_pandas(df)
                else:
                    yield df
        else:
            # Fall back to loading entire file
            yield self.load(path, format=format, **kwargs)
    
    def save(
        self,
        df: Any,
        path: Union[str, Path],
        format: str = "parquet",
        partition_cols: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            path: Output path
            format: Output format
            partition_cols: Columns to partition by
            **kwargs: Additional format-specific arguments
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            if self.use_gpu:
                df.to_parquet(str(path), partition_cols=partition_cols, **kwargs)
            else:
                df.to_parquet(str(path), partition_cols=partition_cols, **kwargs)
        elif format == "csv":
            df.to_csv(str(path), index=False, **kwargs)
        elif format == "json":
            df.to_json(str(path), **kwargs)
        
        logger.info(f"Saved {len(df):,} rows to {path}")


class InteractionDataLoader:
    """Specialized loader for user-item interaction data."""
    
    def __init__(self, base_loader: Optional[DataLoader] = None):
        self.loader = base_loader or DataLoader()
    
    def load_interactions(
        self,
        path: Union[str, Path],
        user_col: str = "user_id",
        item_col: str = "item_id",
        event_col: str = "event_type",
        timestamp_col: str = "timestamp",
        **kwargs,
    ) -> Any:
        """Load interaction data with standard schema."""
        df = self.loader.load(path, **kwargs)
        
        # Validate required columns
        required = [user_col, item_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Standardize column names
        rename_map = {
            user_col: "user_id",
            item_col: "item_id",
        }
        if event_col in df.columns:
            rename_map[event_col] = "event_type"
        if timestamp_col in df.columns:
            rename_map[timestamp_col] = "timestamp"
        
        df = df.rename(columns=rename_map)
        
        return df
    
    def split_by_time(
        self,
        df: Any,
        train_days: int = 30,
        val_days: int = 7,
        test_days: int = 7,
        timestamp_col: str = "timestamp",
    ) -> tuple:
        """Split data by time for train/val/test."""
        if timestamp_col not in df.columns:
            raise ValueError(f"Column {timestamp_col} not found")
        
        # Get max timestamp
        if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
            max_ts = df[timestamp_col].max()
        else:
            max_ts = df[timestamp_col].max()
        
        # Calculate split points
        test_start = max_ts - pd.Timedelta(days=test_days)
        val_start = test_start - pd.Timedelta(days=val_days)
        train_start = val_start - pd.Timedelta(days=train_days)
        
        # Split
        train_mask = (df[timestamp_col] >= train_start) & (df[timestamp_col] < val_start)
        val_mask = (df[timestamp_col] >= val_start) & (df[timestamp_col] < test_start)
        test_mask = df[timestamp_col] >= test_start
        
        train_df = df[train_mask]
        val_df = df[val_mask]
        test_df = df[test_mask]
        
        logger.info(
            f"Split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}"
        )
        
        return train_df, val_df, test_df


class FeatureDataLoader:
    """Loader for user/item feature data."""
    
    def __init__(self, base_loader: Optional[DataLoader] = None):
        self.loader = base_loader or DataLoader()
    
    def load_user_features(
        self,
        path: Union[str, Path],
        user_col: str = "user_id",
        **kwargs,
    ) -> Dict[str, Any]:
        """Load user features into a lookup dict."""
        df = self.loader.load(path, **kwargs)
        
        # Convert to dict for fast lookup
        if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        
        df = df.set_index(user_col)
        return df.to_dict(orient='index')
    
    def load_item_features(
        self,
        path: Union[str, Path],
        item_col: str = "item_id",
        **kwargs,
    ) -> Dict[str, Any]:
        """Load item features into a lookup dict."""
        df = self.loader.load(path, **kwargs)
        
        if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        
        df = df.set_index(item_col)
        return df.to_dict(orient='index')
