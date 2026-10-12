"""Text Normalization for TTS."""

import re
from typing import Dict


class TextNormalizer:
    """Normalize text for TTS synthesis."""
    
    def __init__(self):
        self.abbreviations = {
            "Mr.": "Mister", "Mrs.": "Misses", "Ms.": "Miss",
            "Dr.": "Doctor", "Prof.": "Professor",
            "St.": "Street", "Ave.": "Avenue", "Blvd.": "Boulevard",
            "vs.": "versus", "etc.": "etcetera", "e.g.": "for example",
        }
        
        self.number_words = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        }
    
    def normalize(self, text: str) -> str:
        text = self._expand_abbreviations(text)
        text = self._normalize_numbers(text)
        text = self._normalize_symbols(text)
        return text.strip()
    
    def _expand_abbreviations(self, text: str) -> str:
        for abbr, expansion in self.abbreviations.items():
            text = text.replace(abbr, expansion)
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        def replace_number(match):
            num = int(match.group(0))
            if num < 10:
                return self.number_words.get(num, str(num))
            return match.group(0)
        return re.sub(r'\b\d+\b', replace_number, text)
    
    def _normalize_symbols(self, text: str) -> str:
        replacements = {
            "%": " percent", "&": " and ", "@": " at ",
            "$": " dollars ", "€": " euros ", "£": " pounds ",
        }
        for sym, word in replacements.items():
            text = text.replace(sym, word)
        return text
