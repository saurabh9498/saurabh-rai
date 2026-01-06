# Examples

Ready-to-run example scripts demonstrating common use cases for the Recommendation System.

## Available Examples

### 1. Basic Recommendation (`basic_recommendation.py`)

Get personalized recommendations for a single user.

```bash
# Basic usage
python examples/basic_recommendation.py --user-id 12345

# With custom settings
python examples/basic_recommendation.py \
    --user-id 12345 \
    --num-recs 20 \
    --model-path models/checkpoints/dlrm/best.pt \
    --output recommendations.json
```

**What it demonstrates:**
- Loading trained models
- Two-stage retrieval + ranking pipeline
- Displaying results with metadata

---

### 2. Batch Inference (`batch_inference.py`)

Process recommendations for multiple users efficiently.

```bash
# From CSV file
python examples/batch_inference.py \
    --input data/user_ids.csv \
    --output recommendations.parquet

# From comma-separated list
python examples/batch_inference.py \
    --user-ids 1,2,3,4,5,6,7,8,9,10 \
    --output results.json

# With GPU acceleration
python examples/batch_inference.py \
    --input users.csv \
    --output recs.parquet \
    --device cuda \
    --batch-size 128
```

**What it demonstrates:**
- Large-scale batch processing
- GPU acceleration
- Multiple output formats (JSON, Parquet, CSV)
- Progress tracking and throughput metrics

---

### 3. A/B Test Setup (`ab_test_setup.py`)

Configure and manage A/B tests between models.

```bash
# Create new experiment
python examples/ab_test_setup.py \
    --create \
    --name "two_tower_v2_test" \
    --treatment-model models/checkpoints/two_tower/v2.pt \
    --traffic-split 0.2

# Check status
python examples/ab_test_setup.py \
    --status \
    --experiment-id exp_12345

# Analyze results
python examples/ab_test_setup.py \
    --analyze \
    --experiment-id exp_12345 \
    --output analysis.json

# Stop experiment
python examples/ab_test_setup.py \
    --stop \
    --experiment-id exp_12345
```

**What it demonstrates:**
- Creating controlled experiments
- Traffic allocation
- Statistical significance testing
- Automated recommendations

---

## Prerequisites

Before running examples, ensure you have:

1. **Installed dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Trained models available:**
   ```bash
   # Or download pre-trained
   python scripts/train.py --config configs/training_config.yaml
   ```

3. **Sample data generated:**
   ```bash
   python scripts/generate_sample_data.py
   ```

## Quick Start

```bash
# Generate sample data
python scripts/generate_sample_data.py

# Train a model (or use pre-trained)
python scripts/train.py --epochs 5 --save-dir models/checkpoints/

# Run basic example
python examples/basic_recommendation.py --user-id 1
```

## Output Formats

All examples support multiple output formats:

| Format | Extension | Use Case |
|--------|-----------|----------|
| JSON | `.json` | Human-readable, APIs |
| Parquet | `.parquet` | Analytics, large datasets |
| CSV | `.csv` | Spreadsheets, simple analysis |

## Integration Examples

### Python API

```python
from examples.basic_recommendation import get_recommendations

# Get recommendations programmatically
recs = get_recommendations(user_id=12345, num_recommendations=10)
for rec in recs:
    print(f"Item: {rec['item_id']}, Score: {rec['score']:.4f}")
```

### REST API

```bash
# Start the API server
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

# Get recommendations via API
curl "http://localhost:8000/recommend?user_id=12345&num_recs=10"
```

## Troubleshooting

**Model not found:**
```bash
# Check available models
ls -la models/checkpoints/
```

**CUDA out of memory:**
```bash
# Use smaller batch size or CPU
python examples/batch_inference.py --device cpu --batch-size 16
```

**Import errors:**
```bash
# Run from project root
cd recommendation-system
python examples/basic_recommendation.py --user-id 1
```
