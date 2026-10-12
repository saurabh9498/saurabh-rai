"""
Entity Extraction

Named Entity Recognition for extracting slots from user utterances.
Supports custom entity types, regex patterns, and transformer-based extraction.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted entity."""
    type: str
    value: str
    raw_value: str
    start: int
    end: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "value": self.value,
            "raw_value": self.raw_value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: List[Entity]
    text: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "text": self.text,
        }
    
    def get_entity(self, entity_type: str) -> Optional[Entity]:
        """Get first entity of given type."""
        for entity in self.entities:
            if entity.type == entity_type:
                return entity
        return None
    
    def get_entities(self, entity_type: str) -> List[Entity]:
        """Get all entities of given type."""
        return [e for e in self.entities if e.type == entity_type]
    
    def get_value(self, entity_type: str) -> Optional[str]:
        """Get value of first entity of given type."""
        entity = self.get_entity(entity_type)
        return entity.value if entity else None


class PatternExtractor:
    """
    Rule-based entity extraction using patterns.
    
    Handles common entity types like dates, times, numbers, etc.
    """
    
    def __init__(self):
        self.patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[str, List[Tuple[re.Pattern, callable]]]:
        """Build regex patterns for entity types."""
        return {
            "time": [
                (re.compile(r'\b(\d{1,2}):(\d{2})\s*(am|pm|AM|PM)?\b'), self._parse_time),
                (re.compile(r'\b(\d{1,2})\s*(am|pm|AM|PM)\b'), self._parse_time_simple),
                (re.compile(r'\b(noon|midnight|morning|afternoon|evening|night)\b', re.I), self._parse_time_word),
            ],
            "date": [
                (re.compile(r'\b(today|tomorrow|yesterday)\b', re.I), self._parse_relative_date),
                (re.compile(r'\b(next|this|last)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year)\b', re.I), self._parse_relative_date_complex),
                (re.compile(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b'), self._parse_date_numeric),
                (re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b', re.I), self._parse_date_text),
            ],
            "duration": [
                (re.compile(r'\b(\d+)\s*(second|minute|hour|day|week|month|year)s?\b', re.I), self._parse_duration),
                (re.compile(r'\b(half|quarter)\s*(an?\s+)?(hour|minute)\b', re.I), self._parse_duration_fraction),
            ],
            "number": [
                (re.compile(r'\b(\d+(?:\.\d+)?)\b'), self._parse_number),
                (re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b', re.I), self._parse_number_word),
            ],
            "temperature": [
                (re.compile(r'\b(\d+)\s*°?\s*(fahrenheit|celsius|f|c)\b', re.I), self._parse_temperature),
            ],
            "location": [
                (re.compile(r'\bin\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z]{2})?)\b'), self._parse_location),
            ],
            "email": [
                (re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b'), self._parse_email),
            ],
            "phone": [
                (re.compile(r'\b(?:\+?1[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b'), self._parse_phone),
            ],
        }
    
    def extract(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern, parser in patterns:
                for match in pattern.finditer(text):
                    try:
                        value = parser(match)
                        if value:
                            entity = Entity(
                                type=entity_type,
                                value=value,
                                raw_value=match.group(0),
                                start=match.start(),
                                end=match.end(),
                                confidence=0.9,
                            )
                            entities.append(entity)
                    except Exception as e:
                        logger.debug(f"Failed to parse {entity_type}: {e}")
        
        return entities
    
    def _parse_time(self, match) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3)
        
        if period and period.lower() == 'pm' and hour < 12:
            hour += 12
        elif period and period.lower() == 'am' and hour == 12:
            hour = 0
        
        return f"{hour:02d}:{minute:02d}"
    
    def _parse_time_simple(self, match) -> str:
        hour = int(match.group(1))
        period = match.group(2).lower()
        
        if period == 'pm' and hour < 12:
            hour += 12
        elif period == 'am' and hour == 12:
            hour = 0
        
        return f"{hour:02d}:00"
    
    def _parse_time_word(self, match) -> str:
        word = match.group(1).lower()
        times = {
            "noon": "12:00",
            "midnight": "00:00",
            "morning": "09:00",
            "afternoon": "14:00",
            "evening": "18:00",
            "night": "21:00",
        }
        return times.get(word, "12:00")
    
    def _parse_relative_date(self, match) -> str:
        word = match.group(1).lower()
        today = datetime.now().date()
        
        if word == "today":
            return today.isoformat()
        elif word == "tomorrow":
            return (today + timedelta(days=1)).isoformat()
        elif word == "yesterday":
            return (today - timedelta(days=1)).isoformat()
        
        return today.isoformat()
    
    def _parse_relative_date_complex(self, match) -> str:
        modifier = match.group(1).lower()
        unit = match.group(2).lower()
        today = datetime.now().date()
        
        # Simplified implementation
        if unit in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            target_day = weekdays.index(unit)
            current_day = today.weekday()
            
            if modifier == "next":
                days_ahead = target_day - current_day + 7
            elif modifier == "last":
                days_ahead = target_day - current_day - 7
            else:  # this
                days_ahead = target_day - current_day
                if days_ahead < 0:
                    days_ahead += 7
            
            return (today + timedelta(days=days_ahead)).isoformat()
        
        return today.isoformat()
    
    def _parse_date_numeric(self, match) -> str:
        month = int(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3))
        
        if year < 100:
            year += 2000
        
        return f"{year:04d}-{month:02d}-{day:02d}"
    
    def _parse_date_text(self, match) -> str:
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        
        month = months[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else datetime.now().year
        
        return f"{year:04d}-{month:02d}-{day:02d}"
    
    def _parse_duration(self, match) -> str:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        
        if not unit.endswith('s'):
            unit += 's'
        
        return f"{amount} {unit}"
    
    def _parse_duration_fraction(self, match) -> str:
        fraction = match.group(1).lower()
        unit = match.group(3).lower()
        
        if fraction == "half":
            minutes = 30 if unit == "hour" else 0.5
        else:  # quarter
            minutes = 15 if unit == "hour" else 0.25
        
        return f"{minutes} minutes" if unit == "hour" else f"{minutes} {unit}s"
    
    def _parse_number(self, match) -> str:
        return match.group(1)
    
    def _parse_number_word(self, match) -> str:
        words = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "eleven": "11", "twelve": "12",
        }
        return words.get(match.group(1).lower(), "0")
    
    def _parse_temperature(self, match) -> str:
        value = match.group(1)
        unit = match.group(2).lower()
        
        if unit in ["f", "fahrenheit"]:
            return f"{value}°F"
        else:
            return f"{value}°C"
    
    def _parse_location(self, match) -> str:
        return match.group(1).strip()
    
    def _parse_email(self, match) -> str:
        return match.group(0)
    
    def _parse_phone(self, match) -> str:
        return f"({match.group(1)}) {match.group(2)}-{match.group(3)}"


class TransformerEntityExtractor:
    """
    Transformer-based entity extraction using NER.
    
    Uses pre-trained models for general entities and
    fine-tuned models for domain-specific slots.
    """
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.model_name = model_name
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load NER pipeline."""
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
            )
            logger.info(f"Loaded NER model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}")
    
    def extract(self, text: str) -> List[Entity]:
        """Extract entities using transformer model."""
        if self.pipeline is None:
            return []
        
        results = self.pipeline(text)
        entities = []
        
        for result in results:
            entity = Entity(
                type=result["entity_group"].lower(),
                value=result["word"],
                raw_value=result["word"],
                start=result["start"],
                end=result["end"],
                confidence=result["score"],
            )
            entities.append(entity)
        
        return entities


class EntityExtractor:
    """
    Combined entity extraction service.
    
    Uses pattern matching for structured entities and
    transformer models for general NER.
    """
    
    def __init__(self, use_transformer: bool = True):
        self.pattern_extractor = PatternExtractor()
        self.transformer_extractor = None
        
        if use_transformer:
            self.transformer_extractor = TransformerEntityExtractor()
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all entities from text.
        
        Args:
            text: Input text
            
        Returns:
            ExtractionResult with all extracted entities
        """
        entities = []
        
        # Pattern-based extraction
        pattern_entities = self.pattern_extractor.extract(text)
        entities.extend(pattern_entities)
        
        # Transformer-based extraction
        if self.transformer_extractor:
            transformer_entities = self.transformer_extractor.extract(text)
            
            # Merge, avoiding overlaps
            for new_entity in transformer_entities:
                overlaps = False
                for existing in entities:
                    if (new_entity.start < existing.end and 
                        new_entity.end > existing.start):
                        overlaps = True
                        break
                
                if not overlaps:
                    entities.append(new_entity)
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        
        return ExtractionResult(entities=entities, text=text)
