# Outputs Directory

This directory stores generated outputs from the multi-agent system.

## Contents

- Agent responses and reports
- Generated documents
- Exported conversation histories
- Analysis results

## Subdirectories

```
outputs/
├── reports/          # Generated reports and summaries
├── conversations/    # Exported chat histories
└── artifacts/        # Generated code, documents, etc.
```

## Usage

Outputs are saved here by various components:

```python
from src.utils import token_tracker

# Export usage statistics
token_tracker.export_usage("data/outputs/usage_report.json")
```

```python
# Save agent output
with open("data/outputs/report.md", "w") as f:
    f.write(agent_response)
```

## File Naming Convention

Files follow the pattern: `{type}_{timestamp}_{id}.{ext}`

Examples:
- `report_20240115_abc123.md`
- `conversation_20240115_session01.json`
- `analysis_20240115_task42.json`

## Cleanup

Old outputs can be cleaned up with:
```bash
# Remove outputs older than 30 days
find data/outputs -type f -mtime +30 -delete
```

## Note

Large output files should be git-ignored. Only sample outputs are committed.
