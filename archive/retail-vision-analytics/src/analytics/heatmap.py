"""
Heatmap Generation Module for Retail Vision Analytics.

Generates spatial analytics and visualizations:
- Traffic flow heatmaps
- Dwell time intensity maps
- Path density visualization
- Zone comparison analytics
- Time-based pattern analysis

Optimized for real-time updates with efficient accumulation.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class HeatmapType(Enum):
    """Types of heatmaps."""
    TRAFFIC = "traffic"       # Foot traffic density
    DWELL = "dwell"           # Dwell time intensity
    PATH = "path"             # Movement paths
    ATTENTION = "attention"   # Product attention areas
    CONGESTION = "congestion"  # Congestion points


@dataclass
class HeatmapConfig:
    """Heatmap configuration."""
    
    width: int = 1920
    height: int = 1080
    cell_size: int = 20  # Grid cell size in pixels
    decay_rate: float = 0.95  # Temporal decay (0-1)
    blur_sigma: float = 15.0  # Gaussian blur sigma
    colormap: str = "jet"  # Color palette
    normalize: bool = True
    
    @property
    def grid_width(self) -> int:
        """Number of grid columns."""
        return self.width // self.cell_size
    
    @property
    def grid_height(self) -> int:
        """Number of grid rows."""
        return self.height // self.cell_size


class HeatmapAccumulator:
    """
    Efficient heatmap accumulator for real-time updates.
    
    Features:
    - Grid-based accumulation for memory efficiency
    - Temporal decay for recency weighting
    - Multiple heatmap types
    - Time-windowed statistics
    
    Example:
        >>> heatmap = HeatmapAccumulator(config)
        >>> heatmap.add_point(500, 300, weight=1.0)
        >>> image = heatmap.render()
    """
    
    # Color palettes
    COLORMAPS = {
        "jet": [(0, 0, 128), (0, 0, 255), (0, 255, 255), 
                (255, 255, 0), (255, 0, 0), (128, 0, 0)],
        "hot": [(0, 0, 0), (128, 0, 0), (255, 0, 0),
                (255, 128, 0), (255, 255, 0), (255, 255, 255)],
        "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140),
                    (94, 201, 98), (253, 231, 37)],
        "plasma": [(13, 8, 135), (126, 3, 168), (204, 71, 120),
                   (248, 149, 64), (240, 249, 33)]
    }
    
    def __init__(
        self,
        config: Optional[HeatmapConfig] = None,
        heatmap_type: HeatmapType = HeatmapType.TRAFFIC
    ):
        """
        Initialize heatmap accumulator.
        
        Args:
            config: Heatmap configuration
            heatmap_type: Type of heatmap
        """
        self.config = config or HeatmapConfig()
        self.heatmap_type = heatmap_type
        
        # Grid accumulator
        self._grid = np.zeros(
            (self.config.grid_height, self.config.grid_width),
            dtype=np.float32
        )
        
        # Temporal data
        self._last_update = datetime.now()
        self._point_count = 0
        
        # Time-windowed grids for comparison
        self._hourly_grids: Dict[int, np.ndarray] = {}
        
        logger.info(
            f"HeatmapAccumulator initialized: "
            f"{self.config.grid_width}x{self.config.grid_height} grid, "
            f"type={heatmap_type.value}"
        )
    
    def add_point(
        self,
        x: int,
        y: int,
        weight: float = 1.0,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add a point to the heatmap.
        
        Args:
            x: X coordinate
            y: Y coordinate  
            weight: Point weight (default 1.0)
            timestamp: Point timestamp (for temporal analysis)
        """
        # Convert to grid coordinates
        gx = min(max(0, x // self.config.cell_size), self.config.grid_width - 1)
        gy = min(max(0, y // self.config.cell_size), self.config.grid_height - 1)
        
        # Accumulate
        self._grid[gy, gx] += weight
        self._point_count += 1
        
        # Track hourly patterns
        if timestamp:
            hour = timestamp.hour
            if hour not in self._hourly_grids:
                self._hourly_grids[hour] = np.zeros_like(self._grid)
            self._hourly_grids[hour][gy, gx] += weight
    
    def add_points(
        self,
        points: List[Tuple[int, int]],
        weights: Optional[List[float]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add multiple points efficiently."""
        if weights is None:
            weights = [1.0] * len(points)
        
        for (x, y), w in zip(points, weights):
            self.add_point(x, y, w, timestamp)
    
    def add_path(
        self,
        path: List[Tuple[int, int]],
        weight: float = 1.0
    ) -> None:
        """Add a movement path to the heatmap."""
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # Interpolate points along path
            distance = max(abs(x2 - x1), abs(y2 - y1))
            if distance == 0:
                self.add_point(x1, y1, weight)
                continue
            
            for t in range(distance + 1):
                x = int(x1 + (x2 - x1) * t / distance)
                y = int(y1 + (y2 - y1) * t / distance)
                self.add_point(x, y, weight / (distance + 1))
    
    def apply_decay(self, factor: Optional[float] = None) -> None:
        """Apply temporal decay to the heatmap."""
        decay = factor or self.config.decay_rate
        self._grid *= decay
        
        for grid in self._hourly_grids.values():
            grid *= decay
    
    def get_grid(self, normalized: bool = True) -> np.ndarray:
        """
        Get the raw heatmap grid.
        
        Args:
            normalized: Normalize to 0-1 range
        
        Returns:
            2D numpy array
        """
        grid = self._grid.copy()
        
        if normalized and grid.max() > 0:
            grid = grid / grid.max()
        
        return grid
    
    def render(
        self,
        background: Optional[np.ndarray] = None,
        alpha: float = 0.6,
        colormap: Optional[str] = None
    ) -> np.ndarray:
        """
        Render heatmap as colored image.
        
        Args:
            background: Optional background image to overlay
            alpha: Overlay transparency (0-1)
            colormap: Color palette name
        
        Returns:
            BGR image array
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for rendering")
        
        # Get normalized grid
        grid = self.get_grid(normalized=True)
        
        # Upscale to full resolution
        heatmap = cv2.resize(
            grid,
            (self.config.width, self.config.height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Apply Gaussian blur
        if self.config.blur_sigma > 0:
            ksize = int(self.config.blur_sigma * 4) | 1  # Ensure odd
            heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), self.config.blur_sigma)
        
        # Normalize again after blur
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply colormap
        cmap_name = colormap or self.config.colormap
        if cmap_name in ["jet", "hot", "viridis", "plasma"]:
            # Use OpenCV colormap
            cv_cmap = getattr(cv2, f"COLORMAP_{cmap_name.upper()}", cv2.COLORMAP_JET)
            colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv_cmap)
        else:
            colored = self._apply_custom_colormap(heatmap, cmap_name)
        
        # Overlay on background if provided
        if background is not None:
            # Resize background if needed
            if background.shape[:2] != (self.config.height, self.config.width):
                background = cv2.resize(
                    background,
                    (self.config.width, self.config.height)
                )
            
            # Create mask from heatmap intensity
            mask = (heatmap * alpha).reshape(self.config.height, self.config.width, 1)
            
            # Blend
            result = background.astype(np.float32) * (1 - mask) + \
                     colored.astype(np.float32) * mask
            return result.astype(np.uint8)
        
        return colored
    
    def _apply_custom_colormap(
        self,
        heatmap: np.ndarray,
        colormap: str
    ) -> np.ndarray:
        """Apply custom colormap."""
        colors = self.COLORMAPS.get(colormap, self.COLORMAPS["jet"])
        n_colors = len(colors)
        
        result = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
        
        for i in range(n_colors - 1):
            lower = i / (n_colors - 1)
            upper = (i + 1) / (n_colors - 1)
            
            mask = (heatmap >= lower) & (heatmap < upper)
            
            if mask.any():
                t = (heatmap[mask] - lower) / (upper - lower)
                
                for c in range(3):
                    result[mask, c] = (
                        colors[i][c] * (1 - t) + colors[i + 1][c] * t
                    ).astype(np.uint8)
        
        return result
    
    def get_hotspots(
        self,
        threshold: float = 0.7,
        min_area: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Identify high-traffic hotspots.
        
        Args:
            threshold: Intensity threshold (0-1)
            min_area: Minimum hotspot area in grid cells
        
        Returns:
            List of hotspot regions
        """
        grid = self.get_grid(normalized=True)
        
        # Threshold
        binary = (grid >= threshold).astype(np.uint8)
        
        # Find connected components
        if not CV2_AVAILABLE:
            return self._find_hotspots_simple(grid, threshold)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        hotspots = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_area:
                # Get region bounds
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Calculate intensity
                region = grid[y:y+h, x:x+w]
                mask = labels[y:y+h, x:x+w] == i
                intensity = float(np.mean(region[mask]))
                
                # Convert to pixel coordinates
                hotspots.append({
                    "centroid": (
                        int(centroids[i][0] * self.config.cell_size),
                        int(centroids[i][1] * self.config.cell_size)
                    ),
                    "bounds": (
                        x * self.config.cell_size,
                        y * self.config.cell_size,
                        (x + w) * self.config.cell_size,
                        (y + h) * self.config.cell_size
                    ),
                    "area_cells": int(area),
                    "intensity": round(intensity, 3)
                })
        
        return sorted(hotspots, key=lambda h: h["intensity"], reverse=True)
    
    def _find_hotspots_simple(
        self,
        grid: np.ndarray,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Simple hotspot detection without OpenCV."""
        hotspots = []
        
        for gy in range(grid.shape[0]):
            for gx in range(grid.shape[1]):
                if grid[gy, gx] >= threshold:
                    hotspots.append({
                        "centroid": (
                            gx * self.config.cell_size + self.config.cell_size // 2,
                            gy * self.config.cell_size + self.config.cell_size // 2
                        ),
                        "bounds": (
                            gx * self.config.cell_size,
                            gy * self.config.cell_size,
                            (gx + 1) * self.config.cell_size,
                            (gy + 1) * self.config.cell_size
                        ),
                        "area_cells": 1,
                        "intensity": float(grid[gy, gx])
                    })
        
        return hotspots
    
    def get_flow_vectors(
        self,
        points_history: List[List[Tuple[int, int]]]
    ) -> np.ndarray:
        """
        Calculate flow field from trajectory history.
        
        Args:
            points_history: List of trajectories
        
        Returns:
            Flow vector field (grid_h, grid_w, 2)
        """
        flow = np.zeros(
            (self.config.grid_height, self.config.grid_width, 2),
            dtype=np.float32
        )
        counts = np.zeros(
            (self.config.grid_height, self.config.grid_width),
            dtype=np.float32
        )
        
        for trajectory in points_history:
            if len(trajectory) < 2:
                continue
            
            for i in range(len(trajectory) - 1):
                x1, y1 = trajectory[i]
                x2, y2 = trajectory[i + 1]
                
                gx = x1 // self.config.cell_size
                gy = y1 // self.config.cell_size
                
                if 0 <= gx < self.config.grid_width and 0 <= gy < self.config.grid_height:
                    dx = x2 - x1
                    dy = y2 - y1
                    
                    flow[gy, gx, 0] += dx
                    flow[gy, gx, 1] += dy
                    counts[gy, gx] += 1
        
        # Average
        nonzero = counts > 0
        flow[nonzero, 0] /= counts[nonzero]
        flow[nonzero, 1] /= counts[nonzero]
        
        return flow
    
    def get_hourly_comparison(self) -> Dict[int, Dict[str, Any]]:
        """Compare heatmap intensity across hours."""
        comparison = {}
        
        for hour, grid in self._hourly_grids.items():
            total = float(grid.sum())
            max_val = float(grid.max())
            mean_val = float(grid.mean())
            
            comparison[hour] = {
                "total_traffic": round(total, 2),
                "peak_intensity": round(max_val, 4),
                "mean_intensity": round(mean_val, 4),
                "hotspot_count": len(self.get_hotspots(threshold=0.7))
            }
        
        return comparison
    
    def reset(self) -> None:
        """Reset the heatmap accumulator."""
        self._grid.fill(0)
        self._hourly_grids.clear()
        self._point_count = 0
        self._last_update = datetime.now()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get heatmap statistics."""
        grid = self._grid
        
        return {
            "heatmap_type": self.heatmap_type.value,
            "grid_shape": (self.config.grid_height, self.config.grid_width),
            "total_points": self._point_count,
            "total_accumulated": float(grid.sum()),
            "max_intensity": float(grid.max()),
            "mean_intensity": float(grid.mean()),
            "coverage_ratio": float((grid > 0).sum() / grid.size),
            "hotspot_count": len(self.get_hotspots())
        }
    
    def save(self, filepath: str) -> None:
        """Save heatmap to file."""
        np.savez_compressed(
            filepath,
            grid=self._grid,
            config={
                "width": self.config.width,
                "height": self.config.height,
                "cell_size": self.config.cell_size
            },
            point_count=self._point_count
        )
    
    @classmethod
    def load(cls, filepath: str) -> "HeatmapAccumulator":
        """Load heatmap from file."""
        data = np.load(filepath, allow_pickle=True)
        
        config_dict = data["config"].item()
        config = HeatmapConfig(**config_dict)
        
        instance = cls(config)
        instance._grid = data["grid"]
        instance._point_count = int(data["point_count"])
        
        return instance


class MultiHeatmapManager:
    """
    Manages multiple heatmaps for different analytics.
    
    Features:
    - Multiple heatmap types
    - Synchronized updates
    - Comparative analysis
    """
    
    def __init__(self, config: Optional[HeatmapConfig] = None):
        """Initialize manager."""
        self.config = config or HeatmapConfig()
        
        self.heatmaps: Dict[HeatmapType, HeatmapAccumulator] = {}
        
        # Create default heatmaps
        for htype in [HeatmapType.TRAFFIC, HeatmapType.DWELL, HeatmapType.PATH]:
            self.heatmaps[htype] = HeatmapAccumulator(self.config, htype)
    
    def update_traffic(
        self,
        positions: List[Tuple[int, int]],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update traffic heatmap with current positions."""
        self.heatmaps[HeatmapType.TRAFFIC].add_points(positions, timestamp=timestamp)
    
    def update_dwell(
        self,
        position: Tuple[int, int],
        dwell_seconds: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update dwell heatmap weighted by time."""
        weight = min(dwell_seconds / 60.0, 5.0)  # Cap at 5 minutes
        self.heatmaps[HeatmapType.DWELL].add_point(
            position[0], position[1], weight, timestamp
        )
    
    def update_path(
        self,
        trajectory: List[Tuple[int, int]]
    ) -> None:
        """Update path heatmap with trajectory."""
        self.heatmaps[HeatmapType.PATH].add_path(trajectory)
    
    def apply_decay(self) -> None:
        """Apply decay to all heatmaps."""
        for heatmap in self.heatmaps.values():
            heatmap.apply_decay()
    
    def render_all(
        self,
        background: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Render all heatmaps."""
        return {
            htype.value: heatmap.render(background)
            for htype, heatmap in self.heatmaps.items()
        }
    
    def get_comparison(self) -> Dict[str, Any]:
        """Get comparative statistics across heatmaps."""
        return {
            htype.value: heatmap.get_statistics()
            for htype, heatmap in self.heatmaps.items()
        }
    
    def reset_all(self) -> None:
        """Reset all heatmaps."""
        for heatmap in self.heatmaps.values():
            heatmap.reset()


if __name__ == "__main__":
    # Demo usage
    import random
    logging.basicConfig(level=logging.INFO)
    
    # Create heatmap
    config = HeatmapConfig(width=800, height=600, cell_size=20)
    heatmap = HeatmapAccumulator(config, HeatmapType.TRAFFIC)
    
    # Simulate traffic
    for _ in range(1000):
        # Cluster around certain areas
        cluster = random.choice([
            (200, 150),  # Entrance
            (400, 300),  # Center
            (600, 450),  # Display
        ])
        
        x = cluster[0] + random.gauss(0, 50)
        y = cluster[1] + random.gauss(0, 50)
        
        x = max(0, min(config.width - 1, int(x)))
        y = max(0, min(config.height - 1, int(y)))
        
        heatmap.add_point(x, y)
    
    # Get statistics
    print("Heatmap Statistics:")
    stats = heatmap.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Find hotspots
    print("\nHotspots:")
    hotspots = heatmap.get_hotspots(threshold=0.5)
    for i, hs in enumerate(hotspots[:5]):
        print(f"  {i+1}. {hs['centroid']}: intensity={hs['intensity']:.3f}")
    
    # Render if OpenCV available
    if CV2_AVAILABLE:
        image = heatmap.render()
        print(f"\nRendered heatmap shape: {image.shape}")
        # cv2.imwrite("heatmap.png", image)
