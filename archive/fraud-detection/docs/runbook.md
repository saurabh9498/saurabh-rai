# Operational Runbook

## Overview

This runbook provides procedures for operating the fraud detection system.

## Quick Reference

| Issue | Action | Escalation |
|-------|--------|------------|
| High latency (P99 > 20ms) | Check Redis, scale pods | On-call engineer |
| Model drift alert | Review features, consider retraining | ML engineer |
| High error rate (> 1%) | Check logs, restart pods | On-call engineer |
| Kafka lag | Scale consumers | On-call engineer |
| Redis connection failure | Failover to replica | On-call engineer |

## Health Checks

### API Health

```bash
# Basic health check
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 86400,
  "version": "1.0.0"
}
```

### Readiness Check

```bash
curl http://localhost:8000/ready
```

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

## Common Issues

### 1. High Latency

**Symptoms**:
- P99 latency > 20ms
- Slow API responses
- Timeout errors

**Diagnosis**:
```bash
# Check API metrics
curl http://localhost:8000/metrics | grep latency

# Check Redis latency
redis-cli --latency

# Check pod resources
kubectl top pods -n fraud-detection
```

**Resolution**:
1. Scale API pods: `kubectl scale deployment fraud-detection --replicas=5`
2. Check Redis connection pool exhaustion
3. Review recent code changes
4. Check for memory pressure

### 2. Model Drift Alert

**Symptoms**:
- Drift score > 0.1
- Unexpected score distributions
- Increased false positive rate

**Diagnosis**:
```bash
# Check drift metrics
curl http://localhost:8000/metrics | grep drift

# Review feature distributions
python scripts/evaluate.py --check-drift
```

**Resolution**:
1. Review recent data quality
2. Check feature store freshness
3. Consider model retraining
4. Rollback to previous model version if needed

### 3. High Error Rate

**Symptoms**:
- Error rate > 1%
- 500 errors in logs
- Failed transactions

**Diagnosis**:
```bash
# Check API logs
kubectl logs -f deployment/fraud-detection -n fraud-detection

# Check error metrics
curl http://localhost:8000/metrics | grep error
```

**Resolution**:
1. Check Redis connectivity
2. Verify model is loaded
3. Restart pods: `kubectl rollout restart deployment/fraud-detection`
4. Check recent deployments

### 4. Kafka Consumer Lag

**Symptoms**:
- Growing consumer lag
- Delayed fraud scoring
- Backpressure alerts

**Diagnosis**:
```bash
# Check consumer lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
    --group fraud-detector --describe
```

**Resolution**:
1. Scale consumer pods
2. Increase `max_poll_records`
3. Check for slow downstream services
4. Review partition assignment

### 5. Redis Connection Failure

**Symptoms**:
- Connection errors in logs
- Feature store unavailable
- Fallback to default features

**Diagnosis**:
```bash
# Test Redis connectivity
redis-cli ping

# Check Redis cluster status
redis-cli cluster info
```

**Resolution**:
1. Check Redis pod status
2. Verify network connectivity
3. Failover to replica if primary down
4. Check memory usage

## Deployment Procedures

### Standard Deployment

```bash
# Build and push image
docker build -t fraud-detection:v1.2.0 .
docker push registry/fraud-detection:v1.2.0

# Update deployment
kubectl set image deployment/fraud-detection \
    api=registry/fraud-detection:v1.2.0 \
    -n fraud-detection

# Monitor rollout
kubectl rollout status deployment/fraud-detection -n fraud-detection
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/fraud-detection -n fraud-detection

# Rollback to specific revision
kubectl rollout undo deployment/fraud-detection --to-revision=3
```

### Model Deployment

```bash
# Upload new model
aws s3 cp models/ s3://fraud-models/v1.2.0/ --recursive

# Update model ConfigMap
kubectl create configmap model-config \
    --from-literal=MODEL_VERSION=v1.2.0 \
    -n fraud-detection -o yaml --dry-run=client | kubectl apply -f -

# Restart pods to load new model
kubectl rollout restart deployment/fraud-detection -n fraud-detection
```

## Monitoring Dashboards

### Key Dashboards

1. **API Performance**: Request rate, latency, errors
2. **Model Performance**: Score distribution, decisions
3. **Infrastructure**: CPU, memory, network
4. **Business Metrics**: Fraud rate, false positives

### Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| HighLatency | P99 > 20ms for 5m | Warning |
| HighErrorRate | Error rate > 1% for 5m | Critical |
| ModelDrift | Drift score > 0.1 | Warning |
| LowThroughput | TPS < 100 for 5m | Warning |
| RedisDown | Redis unreachable | Critical |

## Scaling Guidelines

### Horizontal Scaling

| Load (TPS) | API Pods | Redis | Kafka Partitions |
|------------|----------|-------|------------------|
| < 5,000 | 3 | 1 primary + 2 replicas | 6 |
| 5,000 - 20,000 | 5 | Cluster (3 nodes) | 12 |
| 20,000 - 50,000 | 10 | Cluster (6 nodes) | 24 |
| > 50,000 | 20+ | Cluster (9+ nodes) | 48+ |

### Auto-scaling

```yaml
# HPA configuration
minReplicas: 3
maxReplicas: 20
targetCPUUtilization: 70%
targetMemoryUtilization: 80%
```

## Emergency Procedures

### Circuit Breaker Activation

If fraud detection is causing transaction failures:

```bash
# Enable bypass mode (approve all)
kubectl set env deployment/fraud-detection BYPASS_MODE=true

# Disable bypass after resolution
kubectl set env deployment/fraud-detection BYPASS_MODE=false
```

### Complete System Restart

```bash
# Scale down
kubectl scale deployment fraud-detection --replicas=0

# Clear Redis cache if needed
redis-cli FLUSHALL

# Scale up
kubectl scale deployment fraud-detection --replicas=5
```

## Contact Information

| Role | Contact |
|------|---------|
| On-call Engineer | PagerDuty |
| ML Engineer | ml-team@company.com |
| Platform Team | platform@company.com |
