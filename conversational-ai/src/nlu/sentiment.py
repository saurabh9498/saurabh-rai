"""Sentiment Analysis."""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    label: str
    score: float
    
    def to_dict(self):
        return {"label": self.label, "score": self.score}


class SentimentAnalyzer:
    """Sentiment analysis using transformers."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.pipeline = None
        try:
            from transformers import pipeline
            self.pipeline = pipeline("sentiment-analysis", model=model_name)
        except Exception as e:
            logger.warning(f"Sentiment model unavailable: {e}")
    
    def analyze(self, text: str) -> SentimentResult:
        if self.pipeline is None:
            return SentimentResult(label="neutral", score=0.5)
        
        result = self.pipeline(text)[0]
        return SentimentResult(
            label=result["label"].lower(),
            score=result["score"],
        )
