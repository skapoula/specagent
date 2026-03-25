#!/bin/bash
# =============================================================================
# Deploy SpecAgent Dev Environment to k3s
# =============================================================================
# Usage:
#   ./k8s/dev/deploy.sh                    # Deploy with defaults
#   ./k8s/dev/deploy.sh --build            # Build and import image first
#   ./k8s/dev/deploy.sh --password mypass  # Set custom password
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
BUILD_IMAGE=false
PASSWORD="specagent-dev-2024"
GROQ_API_KEY="${GROQ_API_KEY:-}"
NAMESPACE="specagent-dev"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_IMAGE=true
            shift
            ;;
        --password)
            PASSWORD="$2"
            shift 2
            ;;
        --groq-key)
            GROQ_API_KEY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Deploying SpecAgent Dev Environment to k3s"
echo "=============================================="

# Build and import image if requested
if [ "$BUILD_IMAGE" = true ]; then
    echo "📦 Building Docker image..."
    docker build -t specagent-dev:latest -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"
    
    echo "📥 Importing image to k3s..."
    docker save specagent-dev:latest | sudo k3s ctr images import -
fi

# Create namespace
echo "📁 Creating namespace..."
kubectl apply -f "$SCRIPT_DIR/00-namespace.yaml"

# Update secrets with provided values
echo "🔐 Configuring secrets..."
kubectl create secret generic dev-secrets \
    --namespace="$NAMESPACE" \
    --from-literal=CODE_SERVER_PASSWORD="$PASSWORD" \
    --from-literal=GROQ_API_KEY="$GROQ_API_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply deployment
echo "🚢 Deploying..."
kubectl apply -f "$SCRIPT_DIR/10-deployment.yaml"

# Wait for pod to be ready
echo "⏳ Waiting for pod to be ready..."
kubectl wait --namespace="$NAMESPACE" \
    --for=condition=ready pod \
    --selector=app.kubernetes.io/name=specagent-dev \
    --timeout=300s

# Get pod name
POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=specagent-dev -o jsonpath='{.items[0].metadata.name}')

echo ""
echo "=============================================="
echo "✅ Dev environment deployed!"
echo "=============================================="
echo ""
echo "Access options:"
echo ""
echo "1. Port forward (recommended for local access):"
echo "   kubectl port-forward -n $NAMESPACE svc/dev-environment 8443:8443"
echo "   Open: http://localhost:8443"
echo "   Password: $PASSWORD"
echo ""
echo "2. If using Ingress, add to /etc/hosts:"
echo "   <node-ip> dev.specagent.local api.specagent.local phoenix.specagent.local"
echo ""
echo "3. Direct pod access:"
echo "   kubectl exec -it -n $NAMESPACE $POD_NAME -- bash"
echo ""
echo "=============================================="
echo "📁 Repository: https://github.com/spiroskapoulas/specagent"
echo "   (Auto-cloned to /workspace on first start)"
echo ""
echo "   To push initial code from local:"
echo "   git remote add origin https://github.com/spiroskapoulas/specagent.git"
echo "   git push -u origin main"
echo ""
echo "🤖 Claude Code setup (Pro subscription):"
echo "   kubectl exec -it -n $NAMESPACE $POD_NAME -- claude login"
echo "   (Opens browser for OAuth - use the URL it provides)"
echo "=============================================="
echo ""
