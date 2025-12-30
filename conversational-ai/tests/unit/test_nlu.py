"""Unit tests for NLU components."""

import pytest
from src.nlu.intent_classifier import IntentClassifier, IntentConfig, IntentResult
from src.nlu.entity_extractor import EntityExtractor, Entity, PatternExtractor
from src.nlu.pipeline import NLUPipeline


class TestIntentClassifier:
    """Tests for intent classification."""
    
    def test_config_defaults(self):
        config = IntentConfig()
        assert config.num_intents == 50
        assert config.confidence_threshold == 0.7
    
    def test_predict(self):
        config = IntentConfig(
            intent_labels=["greeting", "goodbye", "help", "fallback"]
        )
        classifier = IntentClassifier(config)
        
        result = classifier.predict("hello there")
        
        assert isinstance(result, IntentResult)
        assert result.confidence >= 0
        assert result.confidence <= 1
    
    def test_fallback_on_low_confidence(self):
        config = IntentConfig(
            confidence_threshold=0.99,  # Very high threshold
            intent_labels=["test", "fallback"]
        )
        classifier = IntentClassifier(config)
        
        result = classifier.predict("random gibberish xyz")
        
        # Should fall back due to low confidence
        assert result.is_fallback or result.intent == "fallback"


class TestEntityExtractor:
    """Tests for entity extraction."""
    
    def test_extract_time(self):
        extractor = PatternExtractor()
        
        entities = extractor.extract("Set a timer for 3:30 pm")
        time_entities = [e for e in entities if e.type == "time"]
        
        assert len(time_entities) >= 1
        assert time_entities[0].value == "15:30"
    
    def test_extract_date(self):
        extractor = PatternExtractor()
        
        entities = extractor.extract("Remind me tomorrow")
        date_entities = [e for e in entities if e.type == "date"]
        
        assert len(date_entities) == 1
    
    def test_extract_duration(self):
        extractor = PatternExtractor()
        
        entities = extractor.extract("Wait for 5 minutes")
        duration_entities = [e for e in entities if e.type == "duration"]
        
        assert len(duration_entities) == 1
        assert "5" in duration_entities[0].value


class TestNLUPipeline:
    """Tests for complete NLU pipeline."""
    
    def test_process(self):
        pipeline = NLUPipeline(use_transformer_ner=False, enable_sentiment=False)
        
        result = pipeline.process("What's the weather in New York tomorrow?")
        
        assert result.intent is not None
        assert result.entities is not None
        assert result.text == "What's the weather in New York tomorrow?"
