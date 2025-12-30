# Data Directory

This directory contains datasets for the Recommendation System.

---

## Directory Structure

```
data/
├── README.md                 # This file
├── sample/                   # Sample data for testing
│   ├── users.parquet         # User profiles (10K users)
│   ├── items.parquet         # Item catalog (5K items)
│   └── interactions.parquet  # User-item interactions (500K)
├── raw/                      # Raw data (not tracked in git)
│   ├── users/
│   ├── items/
│   └── events/
├── processed/                # Preprocessed features (not tracked)
│   ├── user_features.parquet
│   ├── item_features.parquet
│   └── training_data.parquet
└── embeddings/               # Pre-computed embeddings (not tracked)
    ├── user_embeddings.npy
    └── item_embeddings.npy
```

---

## Data Schemas

### Users (`users.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Unique user identifier |
| age_bucket | int | Age range (0-5 bucketed) |
| gender | int | Gender encoding (0, 1, 2) |
| country | string | ISO country code |
| signup_date | datetime | Account creation date |
| lifetime_value | float | Customer LTV |
| activity_level | int | Engagement tier (1-5) |
| preferred_categories | list[str] | Top category preferences |
| device_type | string | Primary device |
| subscription_tier | int | Free=0, Premium=1, Enterprise=2 |

### Items (`items.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| item_id | string | Unique item identifier |
| category_l1 | string | Top-level category |
| category_l2 | string | Sub-category |
| category_l3 | string | Leaf category |
| brand | string | Brand name |
| price | float | Current price (USD) |
| avg_rating | float | Average user rating (1-5) |
| review_count | int | Number of reviews |
| created_at | datetime | Item listing date |
| popularity_score | float | Rolling 7-day popularity |
| embedding | list[float] | 128-dim content embedding |

### Interactions (`interactions.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | User identifier |
| item_id | string | Item identifier |
| event_type | string | view, click, add_to_cart, purchase |
| timestamp | datetime | Event timestamp (UTC) |
| session_id | string | Session identifier |
| position | int | Item position in list |
| context_device | string | Device used |
| context_page | string | Page where event occurred |
| label | int | Target (1=positive, 0=negative) |

---

## Data Setup Options

### Option 1: Generate Sample Data (Recommended)

```bash
# Generate synthetic data for development
python scripts/generate_sample_data.py \
    --users 10000 \
    --items 5000 \
    --interactions 500000 \
    --output data/sample/
```

### Option 2: Download Public Datasets

```bash
# MovieLens 25M (for benchmarking)
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip -d data/raw/

# Amazon Reviews (product recommendations)
# See: https://nijianmo.github.io/amazon/index.html
```

### Option 3: Connect to Production Data

```bash
# Configure data source in .env
DATA_SOURCE=bigquery
BIGQUERY_PROJECT=your-project
BIGQUERY_DATASET=recommendation_data

# Run ETL pipeline
python scripts/etl_pipeline.py --source bigquery --target data/processed/
```

---

## Sample Data Statistics

| Dataset | Rows | Size | Description |
|---------|------|------|-------------|
| users.parquet | 10,000 | 1.2 MB | User profiles |
| items.parquet | 5,000 | 3.8 MB | Item catalog |
| interactions.parquet | 500,000 | 18 MB | Training events |

### Data Distribution

**Interactions by Event Type:**
- Views: 70%
- Clicks: 20%
- Add to Cart: 7%
- Purchases: 3%

**User Activity Distribution:**
- Power users (>100 events): 5%
- Active users (20-100 events): 25%
- Casual users (5-20 events): 40%
- Low activity (<5 events): 30%

---

## Feature Engineering

The data pipeline generates these feature sets:

### User Features (52 dimensions)
- Demographics: age, gender, country (one-hot)
- Behavioral: activity level, session frequency, avg session length
- Historical: purchase count, cart abandonment rate, return rate
- Embeddings: 32-dim user embedding from interaction history

### Item Features (89 dimensions)
- Categorical: category hierarchy, brand (hashed)
- Numerical: price, rating, review count, popularity
- Temporal: days since listing, trend velocity
- Content: 128-dim embedding from product description

### Context Features (15 dimensions)
- Time: hour of day, day of week, is_weekend
- Device: device type, browser, OS
- Session: items viewed in session, session duration

---

## Privacy & Compliance

⚠️ **Important:** Sample data is 100% synthetic and contains no real user information.

For production data:
- All PII is hashed using SHA-256
- User IDs are anonymized
- Location data is bucketed to region level
- Data retention: 90 days rolling window
- GDPR/CCPA compliant data processing

---

## Data Quality Checks

Run validation before training:

```bash
# Validate data schemas and quality
python scripts/validate_data.py --path data/sample/

# Expected output:
# ✓ Schema validation passed
# ✓ No null values in required columns
# ✓ Referential integrity: all user_ids exist
# ✓ Referential integrity: all item_ids exist
# ✓ Timestamp ordering validated
# ✓ Label distribution within expected range
```

---

## Updating Sample Data

To regenerate sample data with different parameters:

```bash
# Larger dataset for load testing
python scripts/generate_sample_data.py \
    --users 100000 \
    --items 50000 \
    --interactions 10000000 \
    --output data/large/

# Specific time range
python scripts/generate_sample_data.py \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --output data/sample/
```
