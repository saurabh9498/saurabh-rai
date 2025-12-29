#!/usr/bin/env python3
"""
Sample Data Generator for Multi-Agent AI System

Generates realistic sample data for development and testing:
- Knowledge base documents (Markdown, JSON)
- FAQ entries
- Product catalog
- Test queries with expected outputs

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --force  # Overwrite existing
    python scripts/generate_sample_data.py --output /custom/path
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import random


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def create_company_policies() -> str:
    """Generate sample company policies document."""
    return """# Company Policies & Guidelines

## 1. Remote Work Policy

### Eligibility
All full-time employees who have completed their probation period (90 days) are eligible for remote work arrangements.

### Guidelines
- **Core Hours:** Employees must be available between 10 AM - 3 PM in their local timezone
- **Communication:** Respond to messages within 2 hours during work hours
- **Equipment:** Company provides laptop, monitor, and $500 home office stipend
- **Security:** Use VPN for all company resource access

### Approval Process
1. Submit request through HR portal
2. Manager approval required within 5 business days
3. IT security review for sensitive roles
4. HR final confirmation

---

## 2. Time Off Policy

### Paid Time Off (PTO)
| Tenure | Annual PTO Days |
|--------|-----------------|
| 0-2 years | 15 days |
| 2-5 years | 20 days |
| 5+ years | 25 days |

### Sick Leave
- 10 days per year (separate from PTO)
- Doctor's note required for 3+ consecutive days
- Unused sick days do not roll over

### Holidays
Company observes 11 paid holidays per year. See HR calendar for specific dates.

---

## 3. Expense Reimbursement

### Eligible Expenses
- Business travel (flights, hotels, meals)
- Client entertainment (pre-approved)
- Professional development (courses, conferences)
- Home office equipment (with manager approval)

### Submission Process
1. Submit expense within 30 days of purchase
2. Include itemized receipt
3. Manager approval for amounts > $500
4. Finance processes within 2 pay periods

### Per Diem Rates
| Category | Daily Limit |
|----------|-------------|
| Meals (domestic) | $75 |
| Meals (international) | $100 |
| Lodging | Actual cost (reasonable) |

---

## 4. Code of Conduct

### Core Values
- **Integrity:** Act honestly and ethically
- **Respect:** Treat all individuals with dignity
- **Excellence:** Strive for quality in all work
- **Collaboration:** Work together effectively

### Prohibited Behaviors
- Harassment or discrimination
- Conflicts of interest
- Misuse of company resources
- Breach of confidentiality

### Reporting Concerns
- Direct manager
- HR department
- Anonymous ethics hotline: 1-800-ETHICS-1

---

## 5. Information Security

### Password Requirements
- Minimum 12 characters
- Include uppercase, lowercase, numbers, symbols
- Change every 90 days
- No password reuse (last 10 passwords)

### Data Classification
| Level | Examples | Handling |
|-------|----------|----------|
| Public | Marketing materials | No restrictions |
| Internal | Org charts, policies | Internal only |
| Confidential | Customer data, financials | Encrypted, need-to-know |
| Restricted | Trade secrets, M&A | Highest protection |

### Incident Response
Report security incidents immediately to security@company.com or ext. 5555.

---

*Last Updated: January 2024*
*Policy Owner: Human Resources*
*Next Review: January 2025*
"""


def create_technical_docs() -> str:
    """Generate sample technical documentation."""
    return """# Technical Documentation

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
   helm upgrade --install platform ./helm/platform \
     --namespace production \
     --set image.tag=latest
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
"""


def create_product_catalog() -> List[Dict[str, Any]]:
    """Generate sample product catalog."""
    products = [
        {
            "id": "PROD-001",
            "name": "Enterprise Analytics Suite",
            "category": "Analytics",
            "description": "Comprehensive business intelligence platform with real-time dashboards, custom reporting, and AI-powered insights.",
            "pricing": {
                "model": "subscription",
                "tiers": [
                    {"name": "Starter", "price": 499, "period": "month", "users": 5},
                    {"name": "Professional", "price": 1499, "period": "month", "users": 25},
                    {"name": "Enterprise", "price": "custom", "period": "month", "users": "unlimited"}
                ]
            },
            "features": [
                "Real-time dashboards",
                "Custom report builder",
                "50+ data connectors",
                "AI anomaly detection",
                "Scheduled exports",
                "Role-based access control"
            ],
            "integrations": ["Salesforce", "HubSpot", "Snowflake", "BigQuery", "PostgreSQL"],
            "support": "24/7 for Enterprise tier"
        },
        {
            "id": "PROD-002",
            "name": "SecureAuth Platform",
            "category": "Security",
            "description": "Enterprise identity and access management solution with SSO, MFA, and adaptive authentication.",
            "pricing": {
                "model": "per-user",
                "base_price": 3,
                "period": "month",
                "minimum_users": 100
            },
            "features": [
                "Single Sign-On (SSO)",
                "Multi-factor authentication",
                "Adaptive risk-based auth",
                "Directory sync (AD, LDAP)",
                "SCIM provisioning",
                "Audit logging"
            ],
            "compliance": ["SOC 2 Type II", "ISO 27001", "GDPR", "HIPAA"],
            "support": "Business hours + emergency"
        },
        {
            "id": "PROD-003",
            "name": "CloudSync Storage",
            "category": "Infrastructure",
            "description": "Distributed cloud storage solution with automatic replication, versioning, and global CDN.",
            "pricing": {
                "model": "usage-based",
                "storage_per_gb": 0.023,
                "transfer_per_gb": 0.09,
                "requests_per_10k": 0.005
            },
            "features": [
                "11 nines durability",
                "Cross-region replication",
                "Version history (90 days)",
                "Global CDN",
                "Encryption at rest",
                "Lifecycle policies"
            ],
            "sla": "99.99% availability",
            "support": "24/7 for all tiers"
        },
        {
            "id": "PROD-004",
            "name": "DevOps Automation Hub",
            "category": "Developer Tools",
            "description": "End-to-end CI/CD platform with container orchestration, infrastructure as code, and monitoring.",
            "pricing": {
                "model": "subscription",
                "tiers": [
                    {"name": "Team", "price": 299, "period": "month", "build_minutes": 3000},
                    {"name": "Business", "price": 999, "period": "month", "build_minutes": 15000},
                    {"name": "Enterprise", "price": "custom", "period": "month", "build_minutes": "unlimited"}
                ]
            },
            "features": [
                "CI/CD pipelines",
                "Container registry",
                "Kubernetes deployment",
                "Infrastructure as Code",
                "Secret management",
                "Deployment approvals"
            ],
            "integrations": ["GitHub", "GitLab", "Bitbucket", "Jira", "Slack"],
            "support": "Community (Team), Priority (Business+)"
        },
        {
            "id": "PROD-005",
            "name": "AI Document Processor",
            "category": "AI/ML",
            "description": "Intelligent document processing with OCR, entity extraction, and automated classification.",
            "pricing": {
                "model": "per-document",
                "price_per_page": 0.01,
                "volume_discounts": [
                    {"threshold": 10000, "discount": 0.10},
                    {"threshold": 100000, "discount": 0.25},
                    {"threshold": 1000000, "discount": 0.40}
                ]
            },
            "features": [
                "OCR (50+ languages)",
                "Named entity recognition",
                "Document classification",
                "Table extraction",
                "Signature detection",
                "Custom model training"
            ],
            "accuracy": "99.2% OCR accuracy",
            "processing_speed": "< 2 seconds per page"
        }
    ]
    return products


def create_faq() -> List[Dict[str, str]]:
    """Generate sample FAQ entries."""
    faqs = [
        {
            "id": "FAQ-001",
            "category": "Account",
            "question": "How do I reset my password?",
            "answer": "To reset your password: 1) Click 'Forgot Password' on the login page, 2) Enter your email address, 3) Check your inbox for a reset link (valid for 24 hours), 4) Click the link and create a new password meeting our security requirements (12+ characters, mixed case, numbers, symbols)."
        },
        {
            "id": "FAQ-002",
            "category": "Account",
            "question": "How do I enable two-factor authentication (2FA)?",
            "answer": "Enable 2FA in Settings > Security > Two-Factor Authentication. You can choose between authenticator app (recommended), SMS, or hardware key. We recommend using an authenticator app like Google Authenticator or Authy for the best security."
        },
        {
            "id": "FAQ-003",
            "category": "Billing",
            "question": "What payment methods do you accept?",
            "answer": "We accept all major credit cards (Visa, Mastercard, American Express), PayPal, and wire transfers for annual enterprise contracts. All payments are processed securely through our PCI-DSS compliant payment system."
        },
        {
            "id": "FAQ-004",
            "category": "Billing",
            "question": "Can I get a refund?",
            "answer": "We offer a 30-day money-back guarantee for new subscriptions. To request a refund, contact support@company.com with your account details. Refunds are processed within 5-7 business days. Note: Refunds are not available for usage-based charges or after the 30-day period."
        },
        {
            "id": "FAQ-005",
            "category": "Billing",
            "question": "How do I upgrade or downgrade my plan?",
            "answer": "Go to Settings > Subscription > Change Plan. Upgrades take effect immediately with prorated billing. Downgrades take effect at the next billing cycle. Enterprise customers should contact their account manager for plan changes."
        },
        {
            "id": "FAQ-006",
            "category": "Technical",
            "question": "What are the system requirements?",
            "answer": "Our web application works on all modern browsers (Chrome, Firefox, Safari, Edge - latest 2 versions). For desktop apps: Windows 10+, macOS 11+, or Ubuntu 20.04+. Mobile apps require iOS 14+ or Android 10+. Recommended: 4GB RAM, stable internet connection."
        },
        {
            "id": "FAQ-007",
            "category": "Technical",
            "question": "How do I integrate with your API?",
            "answer": "1) Generate an API key in Settings > Developer > API Keys. 2) Review our API documentation at docs.company.com/api. 3) Use our official SDKs (Python, JavaScript, Java, Go) or make direct REST calls. 4) Test in our sandbox environment before production. Rate limits: 1000 requests/minute for standard plans."
        },
        {
            "id": "FAQ-008",
            "category": "Technical",
            "question": "Do you support Single Sign-On (SSO)?",
            "answer": "Yes! We support SAML 2.0 and OIDC for SSO integration. SSO is available on Professional and Enterprise plans. We have pre-built integrations with Okta, Azure AD, OneLogin, and Google Workspace. Contact support for custom IdP configurations."
        },
        {
            "id": "FAQ-009",
            "category": "Data",
            "question": "How is my data protected?",
            "answer": "We implement multiple security layers: AES-256 encryption at rest, TLS 1.3 in transit, SOC 2 Type II certified infrastructure, regular penetration testing, 24/7 security monitoring. Data is stored in ISO 27001 certified data centers with geographic redundancy."
        },
        {
            "id": "FAQ-010",
            "category": "Data",
            "question": "Can I export my data?",
            "answer": "Yes, you can export your data anytime via Settings > Data > Export. Formats available: JSON, CSV, or full database backup (Enterprise). Exports are generated within 24 hours and available for download for 7 days. For GDPR data portability requests, contact privacy@company.com."
        },
        {
            "id": "FAQ-011",
            "category": "Data",
            "question": "Where is my data stored?",
            "answer": "By default, data is stored in US data centers (AWS us-east-1). Enterprise customers can choose EU (eu-west-1), APAC (ap-southeast-1), or dedicated regions. All locations maintain the same security and compliance standards."
        },
        {
            "id": "FAQ-012",
            "category": "Support",
            "question": "What are your support hours?",
            "answer": "Support availability varies by plan: Starter - Email only, 48hr response. Professional - Email + chat, 24hr response, business hours. Enterprise - 24/7 phone + email + chat, 1hr response for critical issues, dedicated success manager."
        },
        {
            "id": "FAQ-013",
            "category": "Support",
            "question": "How do I report a bug or request a feature?",
            "answer": "Report bugs via Help > Report Issue or email bugs@company.com. Include: steps to reproduce, expected vs actual behavior, screenshots/videos if possible. Feature requests can be submitted at feedback.company.com where you can also vote on existing requests."
        },
        {
            "id": "FAQ-014",
            "category": "Product",
            "question": "Do you offer a free trial?",
            "answer": "Yes! We offer a 14-day free trial with full access to Professional tier features. No credit card required to start. At the end of the trial, you can choose a paid plan or your account will convert to our free Starter tier with limited features."
        },
        {
            "id": "FAQ-015",
            "category": "Product",
            "question": "What's included in the Enterprise plan?",
            "answer": "Enterprise includes: unlimited users, custom integrations, dedicated infrastructure, 99.99% SLA, 24/7 premium support, dedicated success manager, custom training, security review, and flexible payment terms. Contact sales@company.com for a custom quote."
        },
        {
            "id": "FAQ-016",
            "category": "Compliance",
            "question": "Are you GDPR compliant?",
            "answer": "Yes, we are fully GDPR compliant. We offer: Data Processing Agreements (DPA), EU data residency options, data portability exports, right to deletion, privacy impact assessments, and appointed Data Protection Officer. Download our DPA at legal.company.com/dpa."
        },
        {
            "id": "FAQ-017",
            "category": "Compliance",
            "question": "Do you have SOC 2 certification?",
            "answer": "Yes, we maintain SOC 2 Type II certification covering Security, Availability, and Confidentiality trust principles. Our latest audit report is available to customers and prospects under NDA. Contact security@company.com to request a copy."
        },
        {
            "id": "FAQ-018",
            "category": "Compliance",
            "question": "Are you HIPAA compliant?",
            "answer": "Yes, our Enterprise tier is HIPAA compliant. We offer Business Associate Agreements (BAA), dedicated HIPAA-compliant infrastructure, audit logging, and additional security controls. Healthcare customers must be on Enterprise plan with signed BAA."
        },
        {
            "id": "FAQ-019",
            "category": "Integration",
            "question": "Do you integrate with Salesforce?",
            "answer": "Yes! Our native Salesforce integration syncs contacts, accounts, opportunities, and custom objects bidirectionally. Install from Salesforce AppExchange, authenticate with OAuth, and configure field mappings. Real-time sync with conflict resolution included."
        },
        {
            "id": "FAQ-020",
            "category": "Integration",
            "question": "Can I use webhooks?",
            "answer": "Yes, webhooks are available on all paid plans. Configure webhooks in Settings > Developer > Webhooks. We support event types for users, data changes, billing events, and more. Webhooks include retry logic (3 attempts) and signature verification for security."
        }
    ]
    return faqs


def create_test_queries() -> List[Dict[str, Any]]:
    """Generate sample test queries with expected outputs."""
    queries = [
        {
            "id": "Q-001",
            "query": "What is the remote work policy?",
            "expected_agent": "research",
            "expected_sources": ["company_policies.md"],
            "expected_topics": ["remote work", "core hours", "eligibility"],
            "difficulty": "easy"
        },
        {
            "id": "Q-002",
            "query": "How much PTO do I get after 3 years?",
            "expected_agent": "research",
            "expected_sources": ["company_policies.md"],
            "expected_answer_contains": ["20 days"],
            "difficulty": "easy"
        },
        {
            "id": "Q-003",
            "query": "Compare the pricing of Analytics Suite vs DevOps Hub",
            "expected_agent": "analyst",
            "expected_sources": ["product_catalog.json"],
            "expected_output_type": "comparison_table",
            "difficulty": "medium"
        },
        {
            "id": "Q-004",
            "query": "Write a Python function to authenticate with our API",
            "expected_agent": "code",
            "expected_sources": ["technical_docs.md"],
            "expected_output_type": "code",
            "expected_language": "python",
            "difficulty": "medium"
        },
        {
            "id": "Q-005",
            "query": "What security certifications do you have?",
            "expected_agent": "research",
            "expected_sources": ["faq.json", "product_catalog.json"],
            "expected_topics": ["SOC 2", "ISO 27001", "GDPR", "HIPAA"],
            "difficulty": "easy"
        },
        {
            "id": "Q-006",
            "query": "I need to submit an expense for a $750 client dinner. What's the process?",
            "expected_agent": "research",
            "expected_sources": ["company_policies.md"],
            "expected_topics": ["expense", "approval", "manager"],
            "difficulty": "medium"
        },
        {
            "id": "Q-007",
            "query": "Create a deployment script for our microservices",
            "expected_agent": "code",
            "expected_sources": ["technical_docs.md"],
            "expected_output_type": "code",
            "expected_language": "bash",
            "difficulty": "hard"
        },
        {
            "id": "Q-008",
            "query": "Analyze which product tier would be best for a 50-person startup",
            "expected_agent": "analyst",
            "expected_sources": ["product_catalog.json", "faq.json"],
            "expected_output_type": "recommendation",
            "difficulty": "hard"
        },
        {
            "id": "Q-009",
            "query": "How do I reset my password and enable 2FA?",
            "expected_agent": "research",
            "expected_sources": ["faq.json"],
            "expected_topics": ["password reset", "two-factor", "authenticator"],
            "difficulty": "easy"
        },
        {
            "id": "Q-010",
            "query": "Troubleshoot: API calls are timing out intermittently",
            "expected_agent": "research",
            "expected_sources": ["technical_docs.md"],
            "expected_topics": ["latency", "troubleshooting", "monitoring"],
            "difficulty": "medium"
        },
        {
            "id": "Q-011",
            "query": "Generate a summary report of all our product offerings for an investor pitch",
            "expected_agent": "analyst",
            "expected_sources": ["product_catalog.json"],
            "expected_output_type": "report",
            "difficulty": "hard"
        },
        {
            "id": "Q-012",
            "query": "What's the per-page cost for processing 500,000 documents?",
            "expected_agent": "analyst",
            "expected_sources": ["product_catalog.json"],
            "expected_calculation": True,
            "difficulty": "medium"
        },
        {
            "id": "Q-013",
            "query": "Draft an email to request remote work approval from my manager",
            "expected_agent": "research",
            "expected_sources": ["company_policies.md"],
            "expected_output_type": "email_draft",
            "difficulty": "medium"
        },
        {
            "id": "Q-014",
            "query": "Explain the authentication flow with a code example",
            "expected_agent": "code",
            "expected_sources": ["technical_docs.md"],
            "expected_output_type": "code_with_explanation",
            "difficulty": "medium"
        },
        {
            "id": "Q-015",
            "query": "What compliance requirements do we meet for a healthcare customer?",
            "expected_agent": "research",
            "expected_sources": ["faq.json", "product_catalog.json"],
            "expected_topics": ["HIPAA", "BAA", "Enterprise", "security"],
            "difficulty": "medium"
        }
    ]
    return queries


def write_file(path: Path, content: str) -> None:
    """Write content to file, creating directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  ✓ Created: {path}")


def write_json(path: Path, data: Any) -> None:
    """Write JSON data to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Created: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample data for Multi-Agent AI System")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory (default: project data/)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing files")
    args = parser.parse_args()

    # Determine output directory
    if args.output:
        data_dir = args.output
    else:
        data_dir = get_project_root() / "data"

    kb_dir = data_dir / "knowledge_base"
    sample_dir = data_dir / "sample"

    print("=" * 60)
    print("Multi-Agent AI System - Sample Data Generator")
    print("=" * 60)
    print(f"\nOutput directory: {data_dir}")
    print()

    # Check for existing files
    existing_files = list(kb_dir.glob("*")) + list(sample_dir.glob("*"))
    if existing_files and not args.force:
        print("⚠️  Existing data files found. Use --force to overwrite.")
        print("   Existing files:")
        for f in existing_files[:5]:
            print(f"     - {f}")
        if len(existing_files) > 5:
            print(f"     ... and {len(existing_files) - 5} more")
        sys.exit(1)

    # Generate knowledge base documents
    print("\n📚 Generating Knowledge Base Documents...")
    print("-" * 40)
    
    write_file(kb_dir / "company_policies.md", create_company_policies())
    write_file(kb_dir / "technical_docs.md", create_technical_docs())
    write_json(kb_dir / "product_catalog.json", create_product_catalog())
    write_json(kb_dir / "faq.json", create_faq())

    # Generate sample/test data
    print("\n🧪 Generating Test Data...")
    print("-" * 40)
    
    write_json(sample_dir / "test_queries.json", create_test_queries())
    
    # Create a simple test documents file
    test_docs = [
        {
            "id": "DOC-001",
            "title": "Q1 Sales Report",
            "content": "Sales increased by 25% in Q1, driven by enterprise deals.",
            "metadata": {"department": "Sales", "date": "2024-04-01"}
        },
        {
            "id": "DOC-002", 
            "title": "Engineering Roadmap",
            "content": "Key initiatives: Platform modernization, AI features, mobile app v2.",
            "metadata": {"department": "Engineering", "date": "2024-03-15"}
        },
        {
            "id": "DOC-003",
            "title": "Customer Feedback Summary",
            "content": "Top requests: better reporting, Slack integration, faster support.",
            "metadata": {"department": "Product", "date": "2024-03-28"}
        }
    ]
    write_json(sample_dir / "test_documents.json", test_docs)

    # Summary
    print("\n" + "=" * 60)
    print("✅ Sample data generation complete!")
    print("=" * 60)
    print(f"""
Files created:
  Knowledge Base:
    - company_policies.md    (Company policies & guidelines)
    - technical_docs.md      (API & system documentation)
    - product_catalog.json   (5 products with pricing)
    - faq.json               (20 FAQ entries)

  Sample/Test Data:
    - test_queries.json      (15 test queries)
    - test_documents.json    (3 test documents)

Next steps:
  1. Review generated data in {data_dir}
  2. Customize content for your use case
  3. Run ingestion: python -m src.rag.ingestion --source {kb_dir}
  4. Test queries: python -m src.agents.orchestrator --test
""")


if __name__ == "__main__":
    main()
