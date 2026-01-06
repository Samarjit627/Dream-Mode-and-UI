# Axis5 Dream PoC — Infra & Terraform Notes

This document outlines a pragmatic infra plan for deploying the Dream PoC.

## Components
- API (FastAPI on 8000)
- Job queue (Redis + Celery/RQ) — optional for PoC, recommended for prod
- Workers (CPU/GPU) for detection/segmentation/CLIP/OCR
- Object store (S3/MinIO) for images, masks, results
- DB (Postgres) for metadata & caching (optional in PoC)

## Terraform (high level)
- VPC, subnets, security groups
- EKS/GKE cluster + node groups (CPU & GPU)
- Container registry (ECR/GCR)
- S3/GCS buckets with lifecycle policies
- Redis (Elasticache/Memorystore)
- RDS (Postgres) if needed
- IAM roles, Secrets Manager (weights, API keys)

## K8s
- Deploy API + workers as separate deployments
- HPA on CPU/gpu utilization + request latency
- Node pools: on-demand CPU, spot GPU (scale to 0 when idle)
- Ingress + TLS

## Weights management
- Store SAM/CLIP/YOLO weights in object store
- InitContainers to fetch on pod start; warm GPU nodes via DaemonSets if needed

## Observability
- Prometheus/Grafana, logs to CloudWatch/Stackdriver
- Tracing (OTel) optional

## Security & privacy
- Signed URLs for artifacts
- Data retention policy
- Third‑party model/API TOS awareness
