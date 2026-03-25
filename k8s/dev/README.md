# k3s Development Environment

Run your SpecAgent development environment in your home lab k3s cluster with code-server (VS Code in browser) and Claude Code pre-installed.

**Repository**: [github.com/spiroskapoulas/specagent](https://github.com/spiroskapoulas/specagent)

## Quick Start

```bash
# 1. Set your Groq API key
export GROQ_API_KEY="your-groq-api-key"

# 2. Build image and deploy
./k8s/dev/deploy.sh --build --password "your-password"

# 3. Access via port-forward
kubectl port-forward -n specagent-dev svc/dev-environment 8443:8443

# 4. Open browser: http://localhost:8443

# 5. The repo is auto-cloned. If empty, push your local files:
cd specagent
git remote add origin https://github.com/spiroskapoulas/specagent.git
git push -u origin main

# 6. Login to Claude Code (Pro subscription uses OAuth)
kubectl exec -it -n specagent-dev deploy/dev-environment -- claude login
# Copy the URL it provides and open in your browser to authenticate
```

## What's Included

| Component | Description |
|-----------|-------------|
| **code-server** | VS Code in browser (port 8443) |
| **Claude Code** | Pre-installed CLI |
| **Python 3.11** | With all project dependencies |
| **Persistent Storage** | 20GB workspace + 10GB cache |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                k3s Cluster                       │
│  ┌───────────────────────────────────────────┐  │
│  │        specagent-dev namespace            │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │     dev-environment pod             │  │  │
│  │  │  ┌─────────────────────────────┐    │  │  │
│  │  │  │  code-server + Claude Code  │    │  │  │
│  │  │  │  :8443 IDE                  │    │  │  │
│  │  │  │  :8000 FastAPI              │    │  │  │
│  │  │  │  :6006 Phoenix              │    │  │  │
│  │  │  └─────────────────────────────┘    │  │  │
│  │  │           │           │             │  │  │
│  │  │    ┌──────┴───┐ ┌─────┴────┐        │  │  │
│  │  │    │workspace │ │  cache   │        │  │  │
│  │  │    │  PVC     │ │  PVC     │        │  │  │
│  │  │    │  20Gi    │ │  10Gi    │        │  │  │
│  │  │    └──────────┘ └──────────┘        │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Access Methods

### 1. Port Forward (Simplest)

```bash
kubectl port-forward -n specagent-dev svc/dev-environment 8443:8443
# Open http://localhost:8443
```

### 2. Ingress (For permanent access)

Edit `k8s/dev/10-deployment.yaml` and update the Ingress hosts:

```yaml
spec:
  rules:
    - host: dev.yourdomain.com  # Change this
```

Add DNS or `/etc/hosts` entry pointing to your k3s node IP.

### 3. NodePort (Alternative)

```bash
kubectl patch svc dev-environment -n specagent-dev -p '{"spec":{"type":"NodePort"}}'
kubectl get svc -n specagent-dev  # Get assigned port
```

## Using Claude Code

Once inside the environment:

```bash
# In code-server terminal or via kubectl exec
claude

# Custom commands available
/implement-node
/implement-retrieval  
/review
```

## Resource Requirements

| Resource | Request | Limit |
|----------|---------|-------|
| Memory | 4Gi | 8Gi |
| CPU | 1 core | 4 cores |
| Storage | 30Gi total | - |

## Customization

### Change Password

```bash
kubectl create secret generic dev-secrets \
  -n specagent-dev \
  --from-literal=CODE_SERVER_PASSWORD="new-password" \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pod
kubectl rollout restart deployment/dev-environment -n specagent-dev
```

### Use Your Own Registry

1. Build and push:
```bash
docker build -t your-registry/specagent-dev:latest -f k8s/dev/Dockerfile .
docker push your-registry/specagent-dev:latest
```

2. Update `10-deployment.yaml`:
```yaml
image: your-registry/specagent-dev:latest
imagePullPolicy: Always
```

### Add GPU Support (Optional)

If your k3s cluster has GPUs:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

## Troubleshooting

```bash
# Check pod status
kubectl get pods -n specagent-dev

# View logs
kubectl logs -n specagent-dev -l app.kubernetes.io/name=specagent-dev

# Shell into pod
kubectl exec -it -n specagent-dev deploy/dev-environment -- bash

# Check storage
kubectl get pvc -n specagent-dev
```

## Cleanup

```bash
kubectl delete namespace specagent-dev
```
