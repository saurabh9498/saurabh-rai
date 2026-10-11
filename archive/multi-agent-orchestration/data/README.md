# Data Directory

This directory contains the knowledge base and sample data for the Multi-Agent AI System.

## Directory Structure

```
data/
├── README.md                 # This file
├── knowledge_base/           # Documents for RAG retrieval
│   ├── company_policies.md   # Sample company policies
│   ├── technical_docs.md     # Technical documentation
│   ├── product_catalog.json  # Product information
│   └── faq.json              # Frequently asked questions
└── sample/                   # Sample data for testing
    ├── queries.json          # Example user queries
    ├── expected_outputs.json # Expected agent responses
    └── test_documents.json   # Test documents for ingestion
```

## Setting Up the Knowledge Base

### Option 1: Use Sample Data (Quick Start)

Generate sample data using the provided script:

```bash
# From project root
python scripts/generate_sample_data.py

# Verify files were created
ls -la data/knowledge_base/
ls -la data/sample/
```

### Option 2: Use Your Own Documents

1. **Supported Formats:**
   - Markdown (.md)
   - PDF (.pdf)
   - Text (.txt)
   - JSON (.json)
   - Word Documents (.docx)

2. **Add Documents:**
   ```bash
   # Copy your documents to knowledge_base
   cp /path/to/your/docs/* data/knowledge_base/
   ```

3. **Ingest Documents:**
   ```bash
   # Run ingestion pipeline
   python -m src.rag.ingestion --source data/knowledge_base/
   ```

### Option 3: Connect to External Sources

Configure external data sources in `config/data_sources.yaml`:

```yaml
sources:
  - type: confluence
    url: https://your-company.atlassian.net
    space_key: DOCS
    
  - type: github
    repo: your-org/documentation
    branch: main
    
  - type: notion
    database_id: your-database-id
```

## Data Requirements

### For Production Use

| Data Type | Recommended Size | Purpose |
|-----------|------------------|---------|
| Knowledge Base | 100+ documents | RAG retrieval accuracy |
| FAQ | 50+ entries | Common query handling |
| Product Catalog | All products | Product-related queries |

### For Development/Testing

The sample data generator creates sufficient data for development:
- 5 knowledge base documents
- 20 FAQ entries
- 10 sample products
- 15 test queries

## Ingestion Pipeline

```
Documents → Chunking → Embedding → Vector Store
              │            │            │
              ▼            ▼            ▼
         512 tokens    OpenAI      ChromaDB
         w/ overlap    ada-002     
```

### Chunking Strategy

- **Chunk Size:** 512 tokens
- **Overlap:** 50 tokens
- **Splitter:** RecursiveCharacterTextSplitter

### Embedding Model

- **Default:** OpenAI text-embedding-ada-002
- **Dimensions:** 1536
- **Alternative:** sentence-transformers/all-MiniLM-L6-v2 (local)

## Environment Variables

Required for data ingestion:

```bash
# .env file
OPENAI_API_KEY=your-api-key          # For embeddings
CHROMA_PERSIST_DIR=./data/chroma     # Vector store location
```

## Data Privacy & Security

⚠️ **Important:**
- Do NOT commit sensitive data to version control
- Use `.gitignore` to exclude proprietary documents
- The `knowledge_base/` samples are for demonstration only
- For production, use encrypted storage for sensitive documents

## Regenerating Sample Data

To regenerate all sample data:

```bash
# Clear existing data
rm -rf data/knowledge_base/*
rm -rf data/sample/*

# Regenerate
python scripts/generate_sample_data.py --force
```

## Troubleshooting

### "No documents found"

```bash
# Check if documents exist
ls data/knowledge_base/

# Regenerate if empty
python scripts/generate_sample_data.py
```

### "Embedding failed"

```bash
# Verify API key
echo $OPENAI_API_KEY

# Test embedding
python -c "from src.rag.embeddings import get_embedding; print(get_embedding('test')[:5])"
```

### "Vector store corrupted"

```bash
# Reset ChromaDB
rm -rf data/chroma/
python -m src.rag.ingestion --source data/knowledge_base/
```
