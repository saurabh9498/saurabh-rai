# Monitoring Configuration

This directory contains monitoring configurations for the Fraud Detection System, including Grafana dashboards and Prometheus alert rules.

## Directory Structure

```
monitoring/
├── README.md                    # This file
├── dashboards/
│   └── fraud_detection.json     # Main Grafana dashboard
└── alerts/
    └── alert_rules.yaml         # Prometheus alerting rules
```

## Dashboards

### Fraud Detection Dashboard

**Location**: `dashboards/fraud_detection.json`

**Panels**:
1. **System Overview**
   - P99 Latency (target: <20ms)
   - Throughput (TPS)
   - Decline Rate
   - Error Rate

2. **Decision Distribution**
   - Decisions over time (APPROVE, STEP_UP, REVIEW, DECLINE)
   - Decision pie chart (1h window)
   - Risk score distribution gauge

3. **Model Performance**
   - Individual model scores (XGBoost, Neural Net, Isolation Forest)
   - Latency percentiles (P50, P95, P99)

### Importing to Grafana

```bash
# Using Grafana API
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @dashboards/fraud_detection.json \
  http://localhost:3000/api/dashboards/db

# Or via Grafana UI:
# 1. Go to Dashboards → Import
# 2. Upload JSON file or paste contents
# 3. Select Prometheus data source
```

## Alert Rules

### Alert Categories

| Category | Alert Count | Severity Range |
|----------|-------------|----------------|
| Availability | 2 | Critical |
| Latency | 2 | Warning/Critical |
| Throughput | 2 | Warning |
| Model Performance | 3 | Warning |
| Infrastructure | 4 | Warning/Critical |
| SLO | 2 | Critical |

### Key Alerts

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| `FraudDetectionServiceDown` | Service unreachable >1m | Critical | Page on-call |
| `FraudDetectionHighLatencyP99` | P99 >20ms for 5m | Warning | Investigate |
| `FraudDetectionHighDeclineRate` | Decline >5% for 30m | Warning | Review model |
| `FraudDetectionSLOLatencyBreach` | <99% under 20ms | Critical | Page on-call |

### Deploying Alert Rules

```bash
# Copy to Prometheus rules directory
cp alerts/alert_rules.yaml /etc/prometheus/rules/

# Reload Prometheus configuration
curl -X POST http://localhost:9090/-/reload

# Or restart Prometheus
systemctl restart prometheus
```

### Alert Routing (Alertmanager)

```yaml
# Example alertmanager.yml configuration
route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'fraud-ops-slack'
  routes:
    - match:
        severity: critical
      receiver: 'fraud-ops-pagerduty'
    - match:
        team: data-science
      receiver: 'data-science-slack'

receivers:
  - name: 'fraud-ops-slack'
    slack_configs:
      - channel: '#fraud-detection-alerts'
        send_resolved: true
  
  - name: 'fraud-ops-pagerduty'
    pagerduty_configs:
      - service_key: '$PAGERDUTY_KEY'
  
  - name: 'data-science-slack'
    slack_configs:
      - channel: '#ml-alerts'
```

## SLOs (Service Level Objectives)

| SLO | Target | Measurement |
|-----|--------|-------------|
| Latency | 99% < 20ms | Rolling 1h window |
| Availability | 99.9% | Rolling 1h window |
| Throughput | >10K TPS | Sustained capacity |
| Error Rate | <1% | Rolling 5m window |

## Metrics Reference

### Application Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_detection_requests_total` | Counter | Total requests processed |
| `fraud_detection_errors_total` | Counter | Total errors |
| `fraud_detection_latency_seconds` | Histogram | Request latency |
| `fraud_detection_decisions_total` | Counter | Decisions by type |
| `fraud_detection_risk_score` | Summary | Risk score distribution |
| `fraud_model_score` | Gauge | Individual model scores |

### Infrastructure Metrics

| Metric | Type | Source |
|--------|------|--------|
| `redis_commands_duration_seconds` | Histogram | Redis exporter |
| `kafka_consumer_group_lag` | Gauge | Kafka exporter |
| `container_memory_usage_bytes` | Gauge | cAdvisor |

## Runbooks

Each alert links to a runbook in the annotations. Runbooks should include:

1. **Alert Context**: What does this alert mean?
2. **Impact Assessment**: What's affected?
3. **Investigation Steps**: How to diagnose?
4. **Remediation**: How to fix?
5. **Escalation**: When and who to escalate to?

Runbook template available at: `docs/runbook.md`
