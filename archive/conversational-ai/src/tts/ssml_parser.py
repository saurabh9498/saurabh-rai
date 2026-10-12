"""SSML Parser for TTS control."""

import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass


@dataclass
class SSMLElement:
    tag: str
    attributes: Dict[str, str]
    text: str


class SSMLParser:
    """Parse SSML markup for TTS synthesis."""
    
    SUPPORTED_TAGS = {"speak", "voice", "prosody", "break", "emphasis", "say-as", "p", "s"}
    
    def parse(self, ssml: str) -> Tuple[str, List[SSMLElement]]:
        try:
            root = ET.fromstring(ssml)
            return self._extract_text_and_elements(root)
        except ET.ParseError:
            return ssml, []
    
    def _extract_text_and_elements(self, element) -> Tuple[str, List[SSMLElement]]:
        text_parts = []
        elements = []
        
        if element.text:
            text_parts.append(element.text)
        
        for child in element:
            if child.tag in self.SUPPORTED_TAGS:
                elements.append(SSMLElement(
                    tag=child.tag,
                    attributes=dict(child.attrib),
                    text=child.text or "",
                ))
            child_text, child_elements = self._extract_text_and_elements(child)
            text_parts.append(child_text)
            elements.extend(child_elements)
            
            if child.tail:
                text_parts.append(child.tail)
        
        return "".join(text_parts), elements
