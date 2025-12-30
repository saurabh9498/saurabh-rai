"""
Intent Classification

Transformer-based intent classification for understanding user intents.
Supports hierarchical intents, confidence calibration, and multi-label classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class IntentConfig:
    """Configuration for intent classification."""
    model_name: str = "distilbert-base-uncased"
    num_intents: int = 50
    intent_labels: List[str] = field(default_factory=list)
    max_length: int = 128
    dropout: float = 0.1
    
    # Confidence settings
    confidence_threshold: float = 0.7
    use_calibration: bool = True
    
    # Multi-label settings
    multi_label: bool = False
    multi_label_threshold: float = 0.5


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: str
    confidence: float
    all_intents: Dict[str, float]
    is_fallback: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "all_intents": self.all_intents,
            "is_fallback": self.is_fallback,
        }


class IntentClassifier(nn.Module):
    """
    BERT-based intent classification model.
    
    Features:
    - Pre-trained transformer encoder
    - Temperature-scaled confidence calibration
    - Hierarchical intent support
    - Fallback detection for OOD queries
    """
    
    def __init__(self, config: IntentConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained encoder
        try:
            from transformers import AutoModel, AutoTokenizer
            self.encoder = AutoModel.from_pretrained(config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        except ImportError:
            logger.warning("Transformers not available, using mock encoder")
            self.encoder = None
            self.tokenizer = None
        
        # Get hidden size
        if self.encoder:
            hidden_size = self.encoder.config.hidden_size
        else:
            hidden_size = 768
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, config.num_intents)
        
        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Logits (batch, num_intents)
        """
        if self.encoder is None:
            # Mock forward pass
            batch_size = input_ids.shape[0]
            return torch.randn(batch_size, self.config.num_intents)
        
        # Encode text
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        # Classify
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict(
        self,
        text: str,
        return_all: bool = False,
    ) -> IntentResult:
        """
        Predict intent for text.
        
        Args:
            text: Input text
            return_all: Return all intent probabilities
            
        Returns:
            IntentResult with prediction
        """
        # Tokenize
        if self.tokenizer:
            inputs = self.tokenizer(
                text,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            # Mock tokenization
            inputs = {
                "input_ids": torch.randint(0, 30000, (1, self.config.max_length)),
                "attention_mask": torch.ones(1, self.config.max_length),
            }
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
            )
            
            # Apply temperature scaling
            if self.config.use_calibration:
                logits = logits / self.temperature
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
        
        # Get top prediction
        max_prob, max_idx = probs[0].max(dim=0)
        confidence = max_prob.item()
        
        # Check for fallback
        is_fallback = confidence < self.config.confidence_threshold
        
        # Get intent label
        if self.config.intent_labels and max_idx < len(self.config.intent_labels):
            intent = self.config.intent_labels[max_idx.item()]
        else:
            intent = f"intent_{max_idx.item()}"
        
        # Build all intents dict
        all_intents = {}
        if return_all or is_fallback:
            for i, prob in enumerate(probs[0].tolist()):
                if self.config.intent_labels and i < len(self.config.intent_labels):
                    label = self.config.intent_labels[i]
                else:
                    label = f"intent_{i}"
                all_intents[label] = prob
        
        if is_fallback:
            intent = "fallback"
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            all_intents=all_intents,
            is_fallback=is_fallback,
        )
    
    def predict_batch(
        self,
        texts: List[str],
    ) -> List[IntentResult]:
        """Predict intents for batch of texts."""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results


class IntentClassifierService:
    """
    Service wrapper for intent classification.
    
    Handles model loading, caching, and inference.
    """
    
    def __init__(
        self,
        config: Optional[IntentConfig] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config or self._load_default_config()
        self.model = IntentClassifier(self.config)
        
        if model_path:
            self._load_weights(model_path)
    
    def _load_default_config(self) -> IntentConfig:
        """Load default intent configuration."""
        return IntentConfig(
            intent_labels=[
                # Information intents
                "get_weather", "get_news", "get_time", "get_date",
                "search_web", "get_definition", "get_translation",
                
                # Control intents
                "play_music", "pause_music", "next_track", "previous_track",
                "set_volume", "set_timer", "set_alarm", "set_reminder",
                "turn_on_device", "turn_off_device",
                
                # Smart home
                "set_thermostat", "control_lights", "lock_door", "check_security",
                
                # Communication
                "send_message", "make_call", "read_messages", "check_email",
                
                # Shopping
                "add_to_cart", "check_order", "track_package", "reorder",
                
                # Travel
                "book_flight", "book_hotel", "get_directions", "check_traffic",
                
                # Conversation
                "greeting", "goodbye", "thank_you", "help", "cancel",
                "yes", "no", "repeat", "what_can_you_do",
                
                # Fallback
                "fallback", "out_of_scope",
            ],
        )
    
    def _load_weights(self, path: str):
        """Load model weights."""
        try:
            state_dict = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded intent classifier from {path}")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
    
    def classify(self, text: str) -> IntentResult:
        """Classify intent for text."""
        return self.model.predict(text, return_all=True)
    
    def classify_batch(self, texts: List[str]) -> List[IntentResult]:
        """Classify intents for batch of texts."""
        return self.model.predict_batch(texts)
    
    def get_intent_list(self) -> List[str]:
        """Get list of supported intents."""
        return self.config.intent_labels.copy()


# Intent taxonomy for hierarchical classification
INTENT_TAXONOMY = {
    "information": [
        "get_weather", "get_news", "get_time", "get_date",
        "search_web", "get_definition", "get_translation",
    ],
    "media_control": [
        "play_music", "pause_music", "next_track", "previous_track",
        "set_volume",
    ],
    "reminders": [
        "set_timer", "set_alarm", "set_reminder",
    ],
    "smart_home": [
        "turn_on_device", "turn_off_device", "set_thermostat",
        "control_lights", "lock_door", "check_security",
    ],
    "communication": [
        "send_message", "make_call", "read_messages", "check_email",
    ],
    "shopping": [
        "add_to_cart", "check_order", "track_package", "reorder",
    ],
    "travel": [
        "book_flight", "book_hotel", "get_directions", "check_traffic",
    ],
    "conversation": [
        "greeting", "goodbye", "thank_you", "help", "cancel",
        "yes", "no", "repeat", "what_can_you_do",
    ],
}


def get_intent_category(intent: str) -> Optional[str]:
    """Get category for an intent."""
    for category, intents in INTENT_TAXONOMY.items():
        if intent in intents:
            return category
    return None
