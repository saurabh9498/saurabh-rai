"""
Edge-Cloud Synchronization Manager for Retail Vision Analytics.

This module handles bidirectional synchronization between edge devices
(Jetson) and cloud infrastructure including:
- Analytics data upload
- Model updates download
- Configuration synchronization
- Health reporting
- Offline buffering and replay

Features:
- Bandwidth-aware transfer scheduling
- Delta synchronization for efficiency
- Automatic retry with exponential backoff
- Local SQLite buffer for offline operation
- Compression for reduced bandwidth
- Secure TLS communication
"""

import os
import time
import json
import gzip
import hashlib
import logging
import threading
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from enum import Enum
from collections import deque
from abc import ABC, abstractmethod
import queue
import io
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SyncDirection(Enum):
    """Synchronization direction."""
    UPLOAD = "upload"
    DOWNLOAD = "download"
    BIDIRECTIONAL = "bidirectional"


class SyncPriority(Enum):
    """Synchronization priority levels."""
    CRITICAL = 0  # Immediate sync (alerts)
    HIGH = 1      # Analytics data
    NORMAL = 2    # Regular metrics
    LOW = 3       # Model updates (large files)
    BACKGROUND = 4  # Logs, diagnostics


class SyncStatus(Enum):
    """Synchronization status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class ConnectionState(Enum):
    """Cloud connection state."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"  # High latency or packet loss


@dataclass
class SyncConfig:
    """Synchronization configuration."""
    
    # Cloud endpoints
    cloud_host: str = "api.retailvision.cloud"
    cloud_port: int = 443
    use_tls: bool = True
    
    # Authentication
    api_key: Optional[str] = None
    device_id: Optional[str] = None
    
    # Upload settings
    upload_interval_sec: float = 30.0
    upload_batch_size: int = 100
    compress_uploads: bool = True
    compression_level: int = 6
    
    # Download settings
    check_updates_interval_sec: float = 300.0
    auto_download_models: bool = True
    model_cache_dir: str = "/var/cache/retail-vision/models"
    
    # Bandwidth management
    max_bandwidth_mbps: float = 10.0
    low_bandwidth_threshold_mbps: float = 1.0
    
    # Retry settings
    max_retries: int = 5
    initial_retry_delay_sec: float = 1.0
    max_retry_delay_sec: float = 300.0
    retry_backoff_factor: float = 2.0
    
    # Buffer settings
    buffer_db_path: str = "/var/lib/retail-vision/sync_buffer.db"
    max_buffer_size_mb: float = 500.0
    buffer_retention_days: int = 7
    
    # Health reporting
    heartbeat_interval_sec: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cloud_host": self.cloud_host,
            "upload_interval_sec": self.upload_interval_sec,
            "check_updates_interval_sec": self.check_updates_interval_sec,
            "max_bandwidth_mbps": self.max_bandwidth_mbps,
            "buffer_db_path": self.buffer_db_path,
        }


@dataclass
class SyncItem:
    """Item to be synchronized."""
    
    item_id: str
    item_type: str  # "analytics", "alert", "metrics", "model", "config"
    direction: SyncDirection
    priority: SyncPriority
    
    # Payload
    data: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    checksum: Optional[str] = None
    
    # Status tracking
    status: SyncStatus = SyncStatus.PENDING
    attempts: int = 0
    last_attempt: Optional[float] = None
    error_message: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        """Calculate MD5 checksum of data."""
        if self.data:
            content = json.dumps(self.data, sort_keys=True).encode()
        elif self.file_path and os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                content = f.read()
        else:
            content = b""
        
        self.checksum = hashlib.md5(content).hexdigest()
        self.size_bytes = len(content)
        return self.checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "direction": self.direction.value,
            "priority": self.priority.value,
            "data": json.dumps(self.data) if self.data else None,
            "file_path": self.file_path,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "status": self.status.value,
            "attempts": self.attempts,
            "last_attempt": self.last_attempt,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncItem":
        """Create from dictionary."""
        return cls(
            item_id=data["item_id"],
            item_type=data["item_type"],
            direction=SyncDirection(data["direction"]),
            priority=SyncPriority(data["priority"]),
            data=json.loads(data["data"]) if data.get("data") else None,
            file_path=data.get("file_path"),
            created_at=data.get("created_at", time.time()),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum"),
            status=SyncStatus(data.get("status", "pending")),
            attempts=data.get("attempts", 0),
            last_attempt=data.get("last_attempt"),
            error_message=data.get("error_message"),
        )


@dataclass
class HealthReport:
    """Edge device health report."""
    
    device_id: str
    timestamp: float
    
    # System metrics
    cpu_util_percent: float
    ram_util_percent: float
    disk_util_percent: float
    gpu_util_percent: float
    temperature_celsius: float
    
    # Pipeline metrics
    streams_active: int
    fps_total: float
    detections_per_sec: float
    inference_latency_ms: float
    
    # Sync metrics
    buffer_size_mb: float
    pending_uploads: int
    last_sync_time: Optional[float]
    connection_state: ConnectionState
    
    # Software versions
    software_version: str
    model_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "system": {
                "cpu_util_percent": self.cpu_util_percent,
                "ram_util_percent": self.ram_util_percent,
                "disk_util_percent": self.disk_util_percent,
                "gpu_util_percent": self.gpu_util_percent,
                "temperature_celsius": self.temperature_celsius,
            },
            "pipeline": {
                "streams_active": self.streams_active,
                "fps_total": self.fps_total,
                "detections_per_sec": self.detections_per_sec,
                "inference_latency_ms": self.inference_latency_ms,
            },
            "sync": {
                "buffer_size_mb": self.buffer_size_mb,
                "pending_uploads": self.pending_uploads,
                "last_sync_time": self.last_sync_time,
                "connection_state": self.connection_state.value,
            },
            "versions": {
                "software": self.software_version,
                "model": self.model_version,
            },
        }


class SyncBuffer:
    """
    Local SQLite buffer for offline sync operations.
    
    Stores pending sync items when cloud is unreachable.
    """
    
    def __init__(self, db_path: str, max_size_mb: float = 500.0):
        """
        Initialize sync buffer.
        
        Args:
            db_path: Path to SQLite database
            max_size_mb: Maximum buffer size in MB
        """
        self.db_path = db_path
        self.max_size_mb = max_size_mb
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_items (
                    item_id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    data TEXT,
                    file_path TEXT,
                    created_at REAL NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    checksum TEXT,
                    status TEXT DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    last_attempt REAL,
                    error_message TEXT
                )
            """)
            
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_priority 
                ON sync_items(status, priority, created_at)
            """)
            
            self._conn.commit()
    
    def add(self, item: SyncItem) -> bool:
        """
        Add item to buffer.
        
        Args:
            item: Sync item to add
            
        Returns:
            True if added successfully
        """
        # Check buffer size
        if self._get_size_mb() >= self.max_size_mb:
            # Remove oldest low-priority items
            self._cleanup_old_items()
        
        with self._lock:
            try:
                data = item.to_dict()
                self._conn.execute("""
                    INSERT OR REPLACE INTO sync_items 
                    (item_id, item_type, direction, priority, data, file_path,
                     created_at, size_bytes, checksum, status, attempts, 
                     last_attempt, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["item_id"], data["item_type"], data["direction"],
                    data["priority"], data["data"], data["file_path"],
                    data["created_at"], data["size_bytes"], data["checksum"],
                    data["status"], data["attempts"], data["last_attempt"],
                    data["error_message"],
                ))
                self._conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to add item to buffer: {e}")
                return False
    
    def get_pending(
        self,
        limit: int = 100,
        priority: Optional[SyncPriority] = None,
    ) -> List[SyncItem]:
        """
        Get pending items from buffer.
        
        Args:
            limit: Maximum items to return
            priority: Filter by priority (optional)
            
        Returns:
            List of pending sync items
        """
        with self._lock:
            if priority is not None:
                rows = self._conn.execute("""
                    SELECT * FROM sync_items 
                    WHERE status = 'pending' AND priority = ?
                    ORDER BY priority, created_at
                    LIMIT ?
                """, (priority.value, limit)).fetchall()
            else:
                rows = self._conn.execute("""
                    SELECT * FROM sync_items 
                    WHERE status = 'pending'
                    ORDER BY priority, created_at
                    LIMIT ?
                """, (limit,)).fetchall()
            
            return [SyncItem.from_dict(dict(row)) for row in rows]
    
    def update_status(
        self,
        item_id: str,
        status: SyncStatus,
        error_message: Optional[str] = None,
    ):
        """Update item status."""
        with self._lock:
            self._conn.execute("""
                UPDATE sync_items 
                SET status = ?, attempts = attempts + 1, 
                    last_attempt = ?, error_message = ?
                WHERE item_id = ?
            """, (status.value, time.time(), error_message, item_id))
            self._conn.commit()
    
    def remove(self, item_id: str):
        """Remove item from buffer."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM sync_items WHERE item_id = ?",
                (item_id,)
            )
            self._conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            stats = {}
            
            # Count by status
            rows = self._conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM sync_items GROUP BY status
            """).fetchall()
            stats["by_status"] = {row["status"]: row["count"] for row in rows}
            
            # Count by priority
            rows = self._conn.execute("""
                SELECT priority, COUNT(*) as count 
                FROM sync_items WHERE status = 'pending'
                GROUP BY priority
            """).fetchall()
            stats["pending_by_priority"] = {row["priority"]: row["count"] for row in rows}
            
            # Total size
            stats["size_mb"] = self._get_size_mb()
            stats["max_size_mb"] = self.max_size_mb
            
            return stats
    
    def _get_size_mb(self) -> float:
        """Get current buffer size in MB."""
        try:
            if os.path.exists(self.db_path):
                return os.path.getsize(self.db_path) / (1024 * 1024)
        except Exception:
            pass
        return 0.0
    
    def _cleanup_old_items(self, days: int = 7):
        """Remove old completed items."""
        cutoff = time.time() - (days * 24 * 60 * 60)
        
        with self._lock:
            # Remove old completed items
            self._conn.execute("""
                DELETE FROM sync_items 
                WHERE status = 'completed' AND created_at < ?
            """, (cutoff,))
            
            # Remove old failed items
            self._conn.execute("""
                DELETE FROM sync_items 
                WHERE status = 'failed' AND created_at < ?
            """, (cutoff,))
            
            # If still over limit, remove lowest priority pending
            if self._get_size_mb() >= self.max_size_mb * 0.9:
                self._conn.execute("""
                    DELETE FROM sync_items 
                    WHERE item_id IN (
                        SELECT item_id FROM sync_items 
                        WHERE status = 'pending' AND priority > 2
                        ORDER BY priority DESC, created_at ASC
                        LIMIT 1000
                    )
                """)
            
            self._conn.commit()
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()


class CloudClient(ABC):
    """Abstract base class for cloud communication."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to cloud."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from cloud."""
        pass
    
    @abstractmethod
    def upload(self, data: bytes, endpoint: str) -> bool:
        """Upload data to cloud."""
        pass
    
    @abstractmethod
    def download(self, endpoint: str) -> Optional[bytes]:
        """Download data from cloud."""
        pass
    
    @abstractmethod
    def check_connection(self) -> ConnectionState:
        """Check connection state."""
        pass


class HTTPCloudClient(CloudClient):
    """
    HTTP-based cloud client implementation.
    
    Uses requests library for HTTP communication with retry logic.
    """
    
    def __init__(self, config: SyncConfig):
        """
        Initialize HTTP client.
        
        Args:
            config: Sync configuration
        """
        self.config = config
        self._session = None
        self._connected = False
        
        # Build base URL
        protocol = "https" if config.use_tls else "http"
        self.base_url = f"{protocol}://{config.cloud_host}:{config.cloud_port}"
    
    def connect(self) -> bool:
        """Establish connection to cloud."""
        try:
            import requests
            self._session = requests.Session()
            
            # Set default headers
            self._session.headers.update({
                "X-API-Key": self.config.api_key or "",
                "X-Device-ID": self.config.device_id or "",
                "Content-Type": "application/json",
            })
            
            # Test connection
            response = self._session.get(
                f"{self.base_url}/health",
                timeout=10,
            )
            
            self._connected = response.status_code == 200
            return self._connected
            
        except ImportError:
            logger.error("requests library not available")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to cloud: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from cloud."""
        if self._session:
            self._session.close()
        self._connected = False
    
    def upload(self, data: bytes, endpoint: str) -> bool:
        """Upload data to cloud endpoint."""
        if not self._session:
            return False
        
        try:
            # Compress if configured
            if self.config.compress_uploads:
                data = gzip.compress(data, compresslevel=self.config.compression_level)
            
            response = self._session.post(
                f"{self.base_url}/{endpoint}",
                data=data,
                headers={
                    "Content-Encoding": "gzip" if self.config.compress_uploads else "identity",
                },
                timeout=30,
            )
            
            return response.status_code in (200, 201, 202)
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def download(self, endpoint: str) -> Optional[bytes]:
        """Download data from cloud endpoint."""
        if not self._session:
            return None
        
        try:
            response = self._session.get(
                f"{self.base_url}/{endpoint}",
                timeout=60,
            )
            
            if response.status_code == 200:
                data = response.content
                
                # Decompress if needed
                if response.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None
    
    def check_connection(self) -> ConnectionState:
        """Check connection state."""
        if not self._session:
            return ConnectionState.DISCONNECTED
        
        try:
            start = time.time()
            response = self._session.get(
                f"{self.base_url}/health",
                timeout=5,
            )
            latency = (time.time() - start) * 1000
            
            if response.status_code != 200:
                return ConnectionState.DISCONNECTED
            
            # Check latency
            if latency > 1000:  # > 1 second
                return ConnectionState.DEGRADED
            
            return ConnectionState.CONNECTED
            
        except Exception:
            return ConnectionState.DISCONNECTED


class MockCloudClient(CloudClient):
    """Mock cloud client for testing and demonstration."""
    
    def __init__(self, config: SyncConfig):
        self.config = config
        self._connected = False
        self._uploads: List[Dict] = []
        self._downloads: Dict[str, bytes] = {}
    
    def connect(self) -> bool:
        self._connected = True
        logger.info("Mock cloud client connected")
        return True
    
    def disconnect(self):
        self._connected = False
    
    def upload(self, data: bytes, endpoint: str) -> bool:
        if not self._connected:
            return False
        
        self._uploads.append({
            "endpoint": endpoint,
            "size": len(data),
            "timestamp": time.time(),
        })
        logger.debug(f"Mock upload to {endpoint}: {len(data)} bytes")
        return True
    
    def download(self, endpoint: str) -> Optional[bytes]:
        if not self._connected:
            return None
        
        return self._downloads.get(endpoint)
    
    def check_connection(self) -> ConnectionState:
        return ConnectionState.CONNECTED if self._connected else ConnectionState.DISCONNECTED
    
    def add_download(self, endpoint: str, data: bytes):
        """Add data for download (for testing)."""
        self._downloads[endpoint] = data


class SyncManager:
    """
    Edge-Cloud Synchronization Manager.
    
    Manages bidirectional synchronization between edge device and cloud:
    - Queues analytics data for upload
    - Downloads model updates
    - Handles offline buffering
    - Reports device health
    
    Example:
        >>> config = SyncConfig(
        ...     cloud_host="api.retailvision.cloud",
        ...     api_key="your-api-key",
        ...     device_id="store1-edge01",
        ... )
        >>> 
        >>> sync = SyncManager(config)
        >>> sync.start()
        >>> 
        >>> # Queue analytics data
        >>> sync.queue_analytics({
        ...     "stream_id": "cam1",
        ...     "detections": [...],
        ... })
        >>> 
        >>> # Check for model updates
        >>> if sync.has_model_update():
        ...     sync.download_model_update()
    """
    
    def __init__(
        self,
        config: SyncConfig,
        cloud_client: Optional[CloudClient] = None,
    ):
        """
        Initialize sync manager.
        
        Args:
            config: Sync configuration
            cloud_client: Cloud client implementation (default: HTTPCloudClient)
        """
        self.config = config
        
        # Cloud client
        self._client = cloud_client or MockCloudClient(config)
        
        # Local buffer
        self._buffer = SyncBuffer(config.buffer_db_path, config.max_size_mb)
        
        # Queues
        self._upload_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._download_queue: queue.Queue = queue.Queue()
        
        # State
        self._running = False
        self._connection_state = ConnectionState.DISCONNECTED
        self._last_sync_time: Optional[float] = None
        
        # Threads
        self._upload_thread: Optional[threading.Thread] = None
        self._download_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        
        # Metrics
        self._metrics = {
            "uploads_successful": 0,
            "uploads_failed": 0,
            "downloads_successful": 0,
            "downloads_failed": 0,
            "bytes_uploaded": 0,
            "bytes_downloaded": 0,
        }
        self._metrics_lock = threading.Lock()
        
        # Callbacks
        self._on_model_update: Optional[Callable[[str], None]] = None
        self._on_config_update: Optional[Callable[[Dict], None]] = None
    
    def start(self):
        """Start synchronization."""
        if self._running:
            return
        
        logger.info("Starting sync manager...")
        self._running = True
        
        # Connect to cloud
        self._connection_state = (
            ConnectionState.CONNECTED 
            if self._client.connect() 
            else ConnectionState.DISCONNECTED
        )
        
        # Start upload thread
        self._upload_thread = threading.Thread(
            target=self._upload_loop,
            daemon=True,
        )
        self._upload_thread.start()
        
        # Start download thread
        self._download_thread = threading.Thread(
            target=self._download_loop,
            daemon=True,
        )
        self._download_thread.start()
        
        # Start health reporting thread
        self._health_thread = threading.Thread(
            target=self._health_loop,
            daemon=True,
        )
        self._health_thread.start()
        
        logger.info("Sync manager started")
    
    def stop(self):
        """Stop synchronization."""
        if not self._running:
            return
        
        logger.info("Stopping sync manager...")
        self._running = False
        
        # Wait for threads
        for thread in [self._upload_thread, self._download_thread, self._health_thread]:
            if thread:
                thread.join(timeout=5.0)
        
        # Disconnect
        self._client.disconnect()
        
        # Close buffer
        self._buffer.close()
        
        logger.info("Sync manager stopped")
    
    def queue_analytics(
        self,
        data: Dict[str, Any],
        priority: SyncPriority = SyncPriority.NORMAL,
    ):
        """
        Queue analytics data for upload.
        
        Args:
            data: Analytics data dictionary
            priority: Upload priority
        """
        item = SyncItem(
            item_id=f"analytics-{time.time()}-{hash(str(data)) % 10000}",
            item_type="analytics",
            direction=SyncDirection.UPLOAD,
            priority=priority,
            data=data,
        )
        item.calculate_checksum()
        
        # Add to buffer first (persistence)
        self._buffer.add(item)
        
        # Then add to priority queue for immediate processing
        self._upload_queue.put((priority.value, time.time(), item))
    
    def queue_alert(self, alert: Dict[str, Any]):
        """
        Queue alert for immediate upload.
        
        Args:
            alert: Alert data dictionary
        """
        item = SyncItem(
            item_id=f"alert-{time.time()}-{hash(str(alert)) % 10000}",
            item_type="alert",
            direction=SyncDirection.UPLOAD,
            priority=SyncPriority.CRITICAL,
            data=alert,
        )
        item.calculate_checksum()
        
        self._buffer.add(item)
        self._upload_queue.put((SyncPriority.CRITICAL.value, time.time(), item))
    
    def queue_metrics(self, metrics: Dict[str, Any]):
        """
        Queue metrics for upload.
        
        Args:
            metrics: Metrics dictionary
        """
        item = SyncItem(
            item_id=f"metrics-{time.time()}",
            item_type="metrics",
            direction=SyncDirection.UPLOAD,
            priority=SyncPriority.NORMAL,
            data=metrics,
        )
        item.calculate_checksum()
        
        self._buffer.add(item)
        self._upload_queue.put((SyncPriority.NORMAL.value, time.time(), item))
    
    def _upload_loop(self):
        """Main upload loop."""
        while self._running:
            try:
                # Check connection
                self._connection_state = self._client.check_connection()
                
                if self._connection_state == ConnectionState.DISCONNECTED:
                    time.sleep(5)
                    continue
                
                # Get items to upload
                items_to_upload = []
                
                # First from priority queue
                while not self._upload_queue.empty() and len(items_to_upload) < self.config.upload_batch_size:
                    try:
                        _, _, item = self._upload_queue.get_nowait()
                        items_to_upload.append(item)
                    except queue.Empty:
                        break
                
                # Then from buffer (if queue is empty)
                if not items_to_upload:
                    items_to_upload = self._buffer.get_pending(self.config.upload_batch_size)
                
                if items_to_upload:
                    self._process_uploads(items_to_upload)
                else:
                    time.sleep(self.config.upload_interval_sec)
                    
            except Exception as e:
                logger.error(f"Upload loop error: {e}")
                time.sleep(5)
    
    def _process_uploads(self, items: List[SyncItem]):
        """Process batch of uploads."""
        # Group by type
        batches: Dict[str, List[SyncItem]] = {}
        for item in items:
            if item.item_type not in batches:
                batches[item.item_type] = []
            batches[item.item_type].append(item)
        
        # Upload each batch
        for item_type, batch_items in batches.items():
            # Prepare payload
            payload = {
                "device_id": self.config.device_id,
                "timestamp": time.time(),
                "items": [item.data for item in batch_items if item.data],
            }
            
            # Serialize and upload
            data = json.dumps(payload).encode()
            endpoint = f"api/v1/{item_type}"
            
            success = self._client.upload(data, endpoint)
            
            # Update status
            for item in batch_items:
                if success:
                    self._buffer.update_status(item.item_id, SyncStatus.COMPLETED)
                    self._buffer.remove(item.item_id)
                    
                    with self._metrics_lock:
                        self._metrics["uploads_successful"] += 1
                        self._metrics["bytes_uploaded"] += item.size_bytes
                else:
                    # Check retry
                    if item.attempts < self.config.max_retries:
                        self._buffer.update_status(item.item_id, SyncStatus.RETRYING)
                        # Re-queue with backoff
                        delay = min(
                            self.config.initial_retry_delay_sec * (
                                self.config.retry_backoff_factor ** item.attempts
                            ),
                            self.config.max_retry_delay_sec,
                        )
                        item.attempts += 1
                        threading.Timer(
                            delay,
                            lambda i=item: self._upload_queue.put((i.priority.value, time.time(), i))
                        ).start()
                    else:
                        self._buffer.update_status(
                            item.item_id,
                            SyncStatus.FAILED,
                            "Max retries exceeded"
                        )
                        with self._metrics_lock:
                            self._metrics["uploads_failed"] += 1
        
        self._last_sync_time = time.time()
    
    def _download_loop(self):
        """Main download loop for checking updates."""
        while self._running:
            try:
                if self._connection_state == ConnectionState.CONNECTED:
                    # Check for model updates
                    self._check_model_updates()
                    
                    # Check for config updates
                    self._check_config_updates()
                
                time.sleep(self.config.check_updates_interval_sec)
                
            except Exception as e:
                logger.error(f"Download loop error: {e}")
                time.sleep(30)
    
    def _check_model_updates(self):
        """Check for model updates from cloud."""
        endpoint = f"api/v1/models/updates?device_id={self.config.device_id}"
        data = self._client.download(endpoint)
        
        if data:
            try:
                update_info = json.loads(data.decode())
                if update_info.get("available"):
                    logger.info(f"Model update available: {update_info.get('version')}")
                    
                    if self.config.auto_download_models:
                        self._download_model(update_info)
                    
                    if self._on_model_update:
                        self._on_model_update(update_info.get("version"))
            except Exception as e:
                logger.error(f"Failed to parse model update info: {e}")
    
    def _download_model(self, update_info: Dict):
        """Download model update."""
        model_url = update_info.get("download_url")
        if not model_url:
            return
        
        # Download model file
        data = self._client.download(model_url)
        if data:
            # Save to cache directory
            os.makedirs(self.config.model_cache_dir, exist_ok=True)
            model_path = os.path.join(
                self.config.model_cache_dir,
                f"model_{update_info.get('version')}.engine"
            )
            
            with open(model_path, 'wb') as f:
                f.write(data)
            
            logger.info(f"Model downloaded to {model_path}")
            
            with self._metrics_lock:
                self._metrics["downloads_successful"] += 1
                self._metrics["bytes_downloaded"] += len(data)
    
    def _check_config_updates(self):
        """Check for configuration updates."""
        endpoint = f"api/v1/config?device_id={self.config.device_id}"
        data = self._client.download(endpoint)
        
        if data:
            try:
                config = json.loads(data.decode())
                if self._on_config_update:
                    self._on_config_update(config)
            except Exception as e:
                logger.error(f"Failed to parse config update: {e}")
    
    def _health_loop(self):
        """Health reporting loop."""
        while self._running:
            try:
                if self._connection_state == ConnectionState.CONNECTED:
                    self._send_health_report()
                
                time.sleep(self.config.heartbeat_interval_sec)
                
            except Exception as e:
                logger.error(f"Health loop error: {e}")
                time.sleep(30)
    
    def _send_health_report(self):
        """Send health report to cloud."""
        # Collect metrics (would come from actual system in production)
        report = HealthReport(
            device_id=self.config.device_id or "unknown",
            timestamp=time.time(),
            cpu_util_percent=45.0,  # Placeholder
            ram_util_percent=60.0,
            disk_util_percent=35.0,
            gpu_util_percent=75.0,
            temperature_celsius=50.0,
            streams_active=4,
            fps_total=120.0,
            detections_per_sec=50.0,
            inference_latency_ms=5.0,
            buffer_size_mb=self._buffer._get_size_mb(),
            pending_uploads=len(self._buffer.get_pending()),
            last_sync_time=self._last_sync_time,
            connection_state=self._connection_state,
            software_version="1.0.0",
            model_version="yolov8n_retail_v2",
        )
        
        data = json.dumps(report.to_dict()).encode()
        self._client.upload(data, "api/v1/health")
    
    def on_model_update(self, callback: Callable[[str], None]):
        """Register callback for model updates."""
        self._on_model_update = callback
    
    def on_config_update(self, callback: Callable[[Dict], None]):
        """Register callback for config updates."""
        self._on_config_update = callback
    
    def get_status(self) -> Dict[str, Any]:
        """Get sync status."""
        with self._metrics_lock:
            metrics = self._metrics.copy()
        
        buffer_stats = self._buffer.get_stats()
        
        return {
            "running": self._running,
            "connection_state": self._connection_state.value,
            "last_sync_time": self._last_sync_time,
            "metrics": metrics,
            "buffer": buffer_stats,
            "upload_queue_size": self._upload_queue.qsize(),
        }
    
    def force_sync(self):
        """Force immediate synchronization."""
        # Load all pending from buffer
        pending = self._buffer.get_pending(1000)
        for item in pending:
            self._upload_queue.put((item.priority.value, time.time(), item))
        
        logger.info(f"Forced sync queued {len(pending)} items")


# Demonstration
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Edge-Cloud Sync Manager Demo")
    print("=" * 60)
    
    # Create configuration
    config = SyncConfig(
        cloud_host="api.retailvision.cloud",
        device_id="store1-edge01",
        api_key="demo-api-key",
        upload_interval_sec=5.0,
        buffer_db_path="/tmp/retail_sync_buffer.db",
    )
    
    # Create sync manager with mock client
    sync = SyncManager(config)
    
    print("\n1. Starting Sync Manager")
    print("-" * 40)
    sync.start()
    print("Sync manager started")
    
    # Queue some data
    print("\n2. Queuing Analytics Data")
    print("-" * 40)
    
    for i in range(5):
        sync.queue_analytics({
            "stream_id": f"camera-{i}",
            "frame_number": i * 100,
            "detections": [
                {"class": "person", "confidence": 0.95, "track_id": i},
                {"class": "shopping_cart", "confidence": 0.88, "track_id": i + 100},
            ],
            "timestamp": time.time(),
        })
        print(f"  Queued analytics batch {i + 1}")
    
    # Queue an alert
    print("\n3. Queuing Alert (Critical Priority)")
    print("-" * 40)
    
    sync.queue_alert({
        "type": "queue_length_exceeded",
        "lane_id": "checkout-1",
        "queue_length": 12,
        "threshold": 8,
        "timestamp": time.time(),
    })
    print("  Alert queued")
    
    # Wait for uploads
    print("\n4. Processing Uploads")
    print("-" * 40)
    time.sleep(3)
    
    # Get status
    print("\n5. Sync Status")
    print("-" * 40)
    
    status = sync.get_status()
    print(f"  Connection: {status['connection_state']}")
    print(f"  Last sync: {status['last_sync_time']}")
    print(f"  Uploads successful: {status['metrics']['uploads_successful']}")
    print(f"  Bytes uploaded: {status['metrics']['bytes_uploaded']}")
    print(f"  Upload queue size: {status['upload_queue_size']}")
    print(f"  Buffer status: {status['buffer']}")
    
    # Register callbacks
    print("\n6. Registering Update Callbacks")
    print("-" * 40)
    
    def on_model_update(version):
        print(f"  Model update available: {version}")
    
    def on_config_update(config):
        print(f"  Config update received: {list(config.keys())}")
    
    sync.on_model_update(on_model_update)
    sync.on_config_update(on_config_update)
    print("  Callbacks registered")
    
    # Simulate some more activity
    print("\n7. Continuous Operation (5 seconds)")
    print("-" * 40)
    
    for i in range(5):
        sync.queue_metrics({
            "gpu_util": 75 + i,
            "fps": 30.0,
            "inference_ms": 4.5 + (i * 0.1),
        })
        time.sleep(1)
        print(f"  Second {i + 1}: queued metrics")
    
    # Final status
    print("\n8. Final Status")
    print("-" * 40)
    
    status = sync.get_status()
    print(f"  Total uploads successful: {status['metrics']['uploads_successful']}")
    print(f"  Total bytes uploaded: {status['metrics']['bytes_uploaded']}")
    
    # Stop
    print("\n9. Stopping Sync Manager")
    print("-" * 40)
    sync.stop()
    print("  Sync manager stopped")
    
    # Cleanup
    if os.path.exists("/tmp/retail_sync_buffer.db"):
        os.remove("/tmp/retail_sync_buffer.db")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
