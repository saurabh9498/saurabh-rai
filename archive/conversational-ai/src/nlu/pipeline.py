"""
NLU Pipeline

Combined intent classification, entity extraction, and sentiment analysis.
Provides a unified interface for natural language understanding.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

from .intent_classifier import IntentClassifierService, IntentResult
from .entity_extractor import EntityExtractor, ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class NLUResult:
    """Complete NLU processing result."""
    intent: IntentResult
    entities: ExtractionResult
    sentiment: Optional[Dict[str, Any]] = None
    text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "entities": self.entities.to_dict(),
            "sentiment": self.sentiment,
            "text": self.text,
        }


class NLUPipeline:
    """
    Complete NLU pipeline combining all understanding components.
    
    Components:
    - Intent Classification
    - Entity Extraction
    - Sentiment Analysis
    """
    
    def __init__(
        self,
        use_transformer_ner: bool = True,
        enable_sentiment: bool = True,
    ):
        self.intent_classifier = IntentClassifierService()
        self.entity_extractor = EntityExtractor(use_transformer=use_transformer_ner)
        self.enable_sentiment = enable_sentiment
        
        if enable_sentiment:
            self._init_sentiment()
    
    def _init_sentiment(self):
        """Initialize sentiment analyzer."""
        try:
            from transformers import pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
        except Exception as e:
            logger.warning(f"Sentiment analysis unavailable: {e}")
            self.sentiment_analyzer = None
    
    def process(self, text: str) -> NLUResult:
        """
        Process text through full NLU pipeline.
        
        Args:
            text: Input text
            
        Returns:
            NLUResult with all understanding components
        """
        # Intent classification
        intent = self.intent_classifier.classify(text)
        
        # Entity extraction
        entities = self.entity_extractor.extract(text)
        
        # Sentiment analysis
        sentiment = None
        if self.enable_sentiment and self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text)[0]
                sentiment = {
                    "label": result["label"].lower(),
                    "score": result["score"],
                }
            except Exception as e:
                logger.debug(f"Sentiment analysis failed: {e}")
        
        return NLUResult(
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            text=text,
        )
    
    def process_batch(self, texts: list) -> list:
        """Process multiple texts."""
        return [self.process(text) for text in texts]
