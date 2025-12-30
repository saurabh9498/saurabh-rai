"""
Kafka Stream Processor

Real-time transaction processing with:
- High-throughput consumption
- Parallel scoring
- Dead letter queue handling
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: str = "localhost:9092"
    input_topic: str = "transactions"
    output_topic: str = "fraud-scores"
    dlq_topic: str = "fraud-dlq"
    consumer_group: str = "fraud-detector"
    auto_offset_reset: str = "latest"
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000


@dataclass
class Transaction:
    """Transaction data structure."""
    transaction_id: str
    card_id: str
    amount: float
    merchant_id: str
    merchant_category: str
    timestamp: datetime
    channel: str
    ip_address: str
    device_id: str
    location: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transaction":
        """Create transaction from dictionary."""
        return cls(
            transaction_id=data["transaction_id"],
            card_id=data["card_id"],
            amount=float(data["amount"]),
            merchant_id=data["merchant_id"],
            merchant_category=data.get("merchant_category", "unknown"),
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
            channel=data.get("channel", "unknown"),
            ip_address=data.get("ip_address", ""),
            device_id=data.get("device_id", ""),
            location=data.get("location"),
            metadata=data.get("metadata"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "card_id": self.card_id,
            "amount": self.amount,
            "merchant_id": self.merchant_id,
            "merchant_category": self.merchant_category,
            "timestamp": self.timestamp.isoformat(),
            "channel": self.channel,
            "ip_address": self.ip_address,
            "device_id": self.device_id,
            "location": self.location,
            "metadata": self.metadata,
        }


class StreamProcessor:
    """
    Kafka-based stream processor for fraud detection.
    
    Consumes transactions, scores them, and produces results.
    """
    
    def __init__(
        self,
        config: Optional[KafkaConfig] = None,
        score_callback: Optional[Callable] = None,
    ):
        self.config = config or KafkaConfig()
        self.score_callback = score_callback
        
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._producer: Optional[AIOKafkaProducer] = None
        self._running = False
        
        # Metrics
        self._messages_processed = 0
        self._errors = 0
        self._total_latency_ms = 0.0
        
    async def start(self):
        """Start the stream processor."""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available, stream processor disabled")
            return
            
        logger.info(f"Starting stream processor...")
        logger.info(f"  Bootstrap servers: {self.config.bootstrap_servers}")
        logger.info(f"  Input topic: {self.config.input_topic}")
        logger.info(f"  Output topic: {self.config.output_topic}")
        
        # Create consumer
        self._consumer = AIOKafkaConsumer(
            self.config.input_topic,
            bootstrap_servers=self.config.bootstrap_servers,
            group_id=self.config.consumer_group,
            auto_offset_reset=self.config.auto_offset_reset,
            max_poll_records=self.config.max_poll_records,
            session_timeout_ms=self.config.session_timeout_ms,
            heartbeat_interval_ms=self.config.heartbeat_interval_ms,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        
        # Create producer
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
        
        await self._consumer.start()
        await self._producer.start()
        
        self._running = True
        logger.info("Stream processor started")
        
    async def stop(self):
        """Stop the stream processor."""
        self._running = False
        
        if self._consumer:
            await self._consumer.stop()
            
        if self._producer:
            await self._producer.stop()
            
        logger.info("Stream processor stopped")
        
    async def run(self):
        """Main processing loop."""
        if not self._consumer or not self._producer:
            await self.start()
            
        logger.info("Starting message processing loop...")
        
        try:
            async for message in self._consumer:
                if not self._running:
                    break
                    
                await self._process_message(message)
                
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            raise
            
    async def _process_message(self, message):
        """Process a single message."""
        import time
        start_time = time.perf_counter()
        
        try:
            # Parse transaction
            transaction = Transaction.from_dict(message.value)
            
            # Score transaction
            if self.score_callback:
                result = await self.score_callback(transaction)
            else:
                result = {"transaction_id": transaction.transaction_id, "risk_score": 0.0}
                
            # Produce result
            await self._producer.send_and_wait(
                self.config.output_topic,
                value=result,
                key=transaction.card_id.encode("utf-8"),
            )
            
            # Update metrics
            latency = (time.perf_counter() - start_time) * 1000
            self._messages_processed += 1
            self._total_latency_ms += latency
            
            if self._messages_processed % 1000 == 0:
                avg_latency = self._total_latency_ms / self._messages_processed
                logger.info(
                    f"Processed {self._messages_processed} messages, "
                    f"avg latency: {avg_latency:.2f}ms"
                )
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self._errors += 1
            
            # Send to dead letter queue
            await self._send_to_dlq(message, str(e))
            
    async def _send_to_dlq(self, message, error: str):
        """Send failed message to dead letter queue."""
        try:
            dlq_message = {
                "original_message": message.value,
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
            }
            
            await self._producer.send_and_wait(
                self.config.dlq_topic,
                value=dlq_message,
            )
            
        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")
            
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        avg_latency = (
            self._total_latency_ms / self._messages_processed
            if self._messages_processed > 0 else 0
        )
        
        return {
            "messages_processed": self._messages_processed,
            "errors": self._errors,
            "avg_latency_ms": avg_latency,
            "error_rate": self._errors / max(self._messages_processed, 1),
        }


class TransactionProducer:
    """Producer for sending transactions to Kafka."""
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        self.config = config or KafkaConfig()
        self._producer: Optional[AIOKafkaProducer] = None
        
    async def start(self):
        """Start the producer."""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available")
            return
            
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        
        await self._producer.start()
        logger.info("Transaction producer started")
        
    async def stop(self):
        """Stop the producer."""
        if self._producer:
            await self._producer.stop()
            
    async def send(self, transaction: Transaction):
        """Send a transaction."""
        if not self._producer:
            await self.start()
            
        await self._producer.send_and_wait(
            self.config.input_topic,
            value=transaction.to_dict(),
            key=transaction.card_id.encode("utf-8"),
        )
        
    async def send_batch(self, transactions: List[Transaction]):
        """Send multiple transactions."""
        if not self._producer:
            await self.start()
            
        batch = self._producer.create_batch()
        
        for txn in transactions:
            batch.append(
                key=txn.card_id.encode("utf-8"),
                value=json.dumps(txn.to_dict()).encode("utf-8"),
                timestamp=None,
            )
            
        await self._producer.send_batch(batch, self.config.input_topic)


class MockStreamProcessor(StreamProcessor):
    """Mock stream processor for testing."""
    
    async def start(self):
        self._running = True
        logger.info("Mock stream processor started")
        
    async def stop(self):
        self._running = False
        
    async def run(self):
        logger.info("Mock stream processor running")
        while self._running:
            await asyncio.sleep(1)
