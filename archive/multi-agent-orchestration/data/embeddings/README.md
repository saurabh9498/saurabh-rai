# Embeddings Directory

This directory stores pre-computed embeddings for the RAG pipeline.

## Contents

- Document embeddings (`.npy`, `.pkl`)
- Embedding indices
- Vector store data (if using local storage)

## Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| NumPy | `.npy` | Raw embedding arrays |
| Pickle | `.pkl` | Embedding objects with metadata |
| FAISS | `.faiss` | FAISS index files |
| ChromaDB | `chroma/` | ChromaDB persistent storage |

## Usage

### Saving Embeddings

```python
import numpy as np
from src.rag.embeddings import EmbeddingGenerator

embedder = EmbeddingGenerator(model="text-embedding-3-small")

# Generate and save embeddings
embeddings = embedder.embed_documents(documents)
np.save("data/embeddings/documents.npy", embeddings)
```

### Loading Embeddings

```python
import numpy as np

embeddings = np.load("data/embeddings/documents.npy")
```

### Using with Vector Store

```python
from src.rag.retriever import VectorRetriever

retriever = VectorRetriever(
    persist_directory="data/embeddings/chroma",
    embedding_model="text-embedding-3-small",
)
```

## Configuration

Embedding settings in `config/rag_config.yaml`:

```yaml
embeddings:
  provider: openai
  model: text-embedding-3-small
  dimensions: 1536
  cache:
    enabled: true
    backend: disk
    directory: ./data/embeddings
```

## Storage Considerations

- Embeddings can be large (1536 dimensions × documents × 4 bytes)
- Consider compression for large datasets
- Use cloud storage (S3, GCS) for production

## Note

Embedding files are git-ignored due to size. Regenerate with:
```bash
python scripts/generate_embeddings.py
```
