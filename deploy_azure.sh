#!/usr/bin/env bash
# Publish the current local state of the app (code + pre-computed files) to Azure.
#   ./deploy_azure.sh
# Builds Dockerfile.azure for Azure's processors, pushes it with a new version tag,
# and switches the running container app to it. Settings and secrets stay as they are.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore

SUBSCRIPTION="Azure for Students"
RG=agemis-rg
APP=agemis
ACR=agemise42b8b
TAG=$(date +%Y%m%d-%H%M)

az account set --subscription "$SUBSCRIPTION"
az acr login --name "$ACR" >/dev/null
docker buildx build --platform linux/amd64 -f Dockerfile.azure -t "$ACR.azurecr.io/agemis:$TAG" --push .
az containerapp update --name "$APP" --resource-group "$RG" --container-name agemis --image "$ACR.azurecr.io/agemis:$TAG" -o none
echo "Deployed version $TAG: https://$(az containerapp show --name "$APP" --resource-group "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
