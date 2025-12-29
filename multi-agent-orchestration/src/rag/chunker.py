"""
Document Chunking Strategies.

This module provides various strategies for splitting documents
into chunks suitable for embedding and retrieval.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import re
import logging


logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED = "fixed"                    # Fixed-size chunks
    RECURSIVE = "recursive"            # Recursive character splitting
    SENTENCE = "sentence"              # Sentence-based splitting
    PARAGRAPH = "paragraph"            # Paragraph-based splitting
    SEMANTIC = "semantic"              # Semantic-based splitting
    MARKDOWN = "markdown"              # Markdown-aware splitting


@dataclass
class ChunkConfig:
    """Configuration for chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]


class DocumentChunker:
    """
    Document chunker with multiple strategies.
    
    Supports:
    - Fixed-size chunking
    - Recursive character splitting
    - Sentence-based splitting
    - Paragraph-based splitting
    - Markdown-aware splitting
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        
        # Strategy mapping
        self._strategies: Dict[ChunkingStrategy, Callable] = {
            ChunkingStrategy.FIXED: self._chunk_fixed,
            ChunkingStrategy.RECURSIVE: self._chunk_recursive,
            ChunkingStrategy.SENTENCE: self._chunk_sentence,
            ChunkingStrategy.PARAGRAPH: self._chunk_paragraph,
            ChunkingStrategy.MARKDOWN: self._chunk_markdown,
        }
    
    def chunk(
        self,
        text: str,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document using the specified strategy.
        
        Args:
            text: The text to chunk
            strategy: Chunking strategy to use
            metadata: Optional metadata to attach to chunks
            **kwargs: Strategy-specific arguments
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not text or not text.strip():
            return []
        
        # Get chunking function
        chunk_func = self._strategies.get(strategy)
        if not chunk_func:
            logger.warning(f"Unknown strategy: {strategy}, falling back to recursive")
            chunk_func = self._chunk_recursive
        
        # Execute chunking
        chunks = chunk_func(text, **kwargs)
        
        # Add metadata to each chunk
        base_metadata = metadata or {}
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "chunk_strategy": strategy.value,
                "chunk_length": len(chunk_text)
            }
            result.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
        
        logger.debug(f"Created {len(result)} chunks using {strategy.value} strategy")
        return result
    
    def _chunk_fixed(self, text: str, **kwargs) -> List[str]:
        """Split text into fixed-size chunks."""
        chunk_size = kwargs.get("chunk_size", self.config.chunk_size)
        overlap = kwargs.get("overlap", self.config.chunk_overlap)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def _chunk_recursive(self, text: str, **kwargs) -> List[str]:
        """
        Recursively split text using a list of separators.
        
        This is the most flexible strategy, trying to split at natural
        boundaries while respecting size limits.
        """
        chunk_size = kwargs.get("chunk_size", self.config.chunk_size)
        overlap = kwargs.get("overlap", self.config.chunk_overlap)
        separators = kwargs.get("separators", self.config.separators)
        
        def split_text(text: str, separators: List[str]) -> List[str]:
            # Base case: text is small enough
            if len(text) <= chunk_size:
                return [text] if text.strip() else []
            
            # Try each separator
            for sep in separators:
                if sep == "":
                    # Last resort: character split
                    return self._chunk_fixed(text, chunk_size=chunk_size, overlap=overlap)
                
                if sep in text:
                    splits = text.split(sep)
                    
                    # Merge small chunks, split large ones
                    chunks = []
                    current_chunk = ""
                    
                    for split in splits:
                        # Would adding this exceed size?
                        potential = current_chunk + sep + split if current_chunk else split
                        
                        if len(potential) <= chunk_size:
                            current_chunk = potential
                        else:
                            # Save current chunk if it exists
                            if current_chunk.strip():
                                chunks.append(current_chunk)
                            
                            # Handle oversized split
                            if len(split) > chunk_size:
                                # Recursively split with next separator
                                sub_chunks = split_text(split, separators[separators.index(sep)+1:])
                                chunks.extend(sub_chunks)
                                current_chunk = ""
                            else:
                                current_chunk = split
                    
                    # Don't forget the last chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk)
                    
                    return chunks
            
            # No separator found, return as is or split
            if len(text) > chunk_size:
                return self._chunk_fixed(text, chunk_size=chunk_size, overlap=overlap)
            return [text]
        
        chunks = split_text(text, separators)
        
        # Add overlap between chunks
        if overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks, overlap)
        
        return chunks
    
    def _chunk_sentence(self, text: str, **kwargs) -> List[str]:
        """Split text by sentences, grouping into chunks."""
        chunk_size = kwargs.get("chunk_size", self.config.chunk_size)
        
        # Split into sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_paragraph(self, text: str, **kwargs) -> List[str]:
        """Split text by paragraphs."""
        chunk_size = kwargs.get("chunk_size", self.config.chunk_size)
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Handle oversized paragraphs
                if len(para) > chunk_size:
                    sub_chunks = self._chunk_sentence(para, chunk_size=chunk_size)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_markdown(self, text: str, **kwargs) -> List[str]:
        """
        Markdown-aware chunking that respects headers and code blocks.
        """
        chunk_size = kwargs.get("chunk_size", self.config.chunk_size)
        
        # Patterns for markdown elements
        header_pattern = r'^(#{1,6})\s+(.+)$'
        code_block_pattern = r'```[\s\S]*?```'
        
        chunks = []
        current_chunk = ""
        current_header = ""
        
        # First, protect code blocks
        code_blocks = re.findall(code_block_pattern, text)
        for i, block in enumerate(code_blocks):
            text = text.replace(block, f"__CODE_BLOCK_{i}__")
        
        # Split by headers
        lines = text.split('\n')
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save current chunk if exists
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                current_header = line
                current_chunk = line
            else:
                potential = current_chunk + "\n" + line if current_chunk else line
                
                if len(potential) <= chunk_size:
                    current_chunk = potential
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = current_header + "\n" + line if current_header else line
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Restore code blocks
        for i, block in enumerate(code_blocks):
            chunks = [chunk.replace(f"__CODE_BLOCK_{i}__", block) for chunk in chunks]
        
        return chunks
    
    def _add_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Add end of previous chunk
                prev_overlap = chunks[i-1][-overlap:]
                chunk = prev_overlap + chunk
            overlapped.append(chunk)
        
        return overlapped
