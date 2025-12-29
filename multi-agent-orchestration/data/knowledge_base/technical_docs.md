# Technical Documentation

## System Architecture Overview

### Microservices Architecture

Our platform consists of the following core services:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   API       │────▶│   Auth      │────▶│  Database   │
│  Gateway    │     │  Service    │     │  Service    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │     │   Payment   │     │  Analytics  │
│  Service    │     │  Service    │     │  Service    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Service Descriptions

#### API Gateway
- **Technology:** Kong / AWS API Gateway
- **Purpose:** Request routing, rate limiting, authentication
- **Port:** 443 (external), 8000 (internal)

#### Authentication Service
- **Technology:** Node.js + Passport.js
- **Database:** Redis (sessions), PostgreSQL (users)
- **Features:** OAuth2, JWT, MFA support

#### User Service
- **Technology:** Python + FastAPI
- **Database:** PostgreSQL
- **Responsibilities:** User CRUD, profiles, preferences

#### Payment Service
- **Technology:** Java + Spring Boot
- **Integrations:** Stripe, PayPal, Wire transfers
- **Compliance:** PCI-DSS Level 1

---

## API Reference

### Authentication

#### POST /api/v1/auth/login

Request:
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

Response (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "dGhpcyBpcyBhIHJlZnJl...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

#### POST /api/v1/auth/refresh

Request:
```json
{
  "refresh_token": "dGhpcyBpcyBhIHJlZnJl..."
}
```

### Users

#### GET /api/v1/users/{user_id}

Headers:
```
Authorization: Bearer <access_token>
```

Response (200 OK):
```json
{
  "id": "usr_123abc",
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2024-01-15T10:30:00Z",
  "role": "member"
}
```

---

## Deployment Guide

### Prerequisites
- Docker 20.10+
- Kubernetes 1.25+
- Helm 3.10+
- AWS CLI configured

### Deployment Steps

1. **Build Images**
   ```bash
   docker build -t api-gateway:latest ./api-gateway
   docker build -t auth-service:latest ./auth-service
   docker build -t user-service:latest ./user-service
   ```

2. **Push to Registry**
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
   docker push $ECR_URI/api-gateway:latest
   ```

3. **Deploy to Kubernetes**
   ```bash
   helm upgrade --install platform ./helm/platform      --namespace production      --set image.tag=latest
   ```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| DATABASE_URL | PostgreSQL connection string | Yes |
| REDIS_URL | Redis connection string | Yes |
| JWT_SECRET | Secret for JWT signing | Yes |
| STRIPE_API_KEY | Stripe API key | Production |

---

## Troubleshooting

### Common Issues

#### "Connection refused" to database
1. Check database pod is running: `kubectl get pods -l app=postgres`
2. Verify service endpoint: `kubectl get svc postgres`
3. Check network policies allow traffic

#### High latency on API calls
1. Check service metrics in Grafana
2. Review slow query logs
3. Verify Redis cache is operational
4. Check for pod resource constraints

#### Authentication failures
1. Verify JWT secret matches across services
2. Check token expiration
3. Review auth service logs: `kubectl logs -l app=auth-service`

---

*Documentation Version: 2.3.0*
*Last Updated: January 2024*
