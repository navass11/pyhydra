#!/usr/bin/env bash
# ============================================================
# Azure Container Instances: SFINCS correlated Manning sims
# Uses existing storage account hydratools / share dockerdata
#
# Requirements:
#   az login           (student subscription)
#   docker login       (Docker Hub, user navass11)
# ============================================================
set -euo pipefail

# ── CONFIG ---------------------------------------------------
DOCKERHUB_USER="navass11"
IMAGE="$DOCKERHUB_USER/sfincs-corr:latest"

SUBSCRIPTION="72a0ae85-d4b9-4457-ba52-c121a9d1b93c"
STORAGE_RG="VisualStudioOnline-3F607BE982BD4B06820771DA8F2FFB4B"
STORAGE_ACCOUNT="hydratools"
FILE_SHARE="dockerdata"

ACI_RG="sfincs-corr-rg"          # new RG just for the containers (easy to delete)

# Allowed regions (sys.regionrestriction policy, 6 StandardCores quota each):
#   francecentral, polandcentral, spaincentral, germanywestcentral, austriaeast
# b0: francecentral 4 vCPU; b1-b9: spread 2 vCPU across other regions
declare -A BATCH_LOCATIONS=(
  [0]="francecentral"
  [1]="polandcentral"     [2]="polandcentral"
  [3]="spaincentral"      [4]="spaincentral"
  [5]="germanywestcentral"[6]="germanywestcentral"
  [7]="austriaeast"       [8]="austriaeast" [9]="austriaeast"
)

N_CONTAINERS=10
SIMS_PER_CONTAINER=10             # 10 × 10 = 100 total
RHO="05"
# ------------------------------------------------------------

az account set --subscription "$SUBSCRIPTION"

echo "=== Step 1: Push image to Docker Hub ==="
docker push "$IMAGE"

echo ""
echo "=== Step 2: Get storage key ==="
STORAGE_KEY=$(az storage account keys list \
  --resource-group "$STORAGE_RG" \
  --account-name   "$STORAGE_ACCOUNT" \
  --query "[0].value" -o tsv)
echo "  Got key for $STORAGE_ACCOUNT"

echo ""
echo "=== Step 3: Create resource group for containers ==="
az group create --name "$ACI_RG" --location "$LOCATION"

echo ""
echo "=== Step 4: Launch $N_CONTAINERS parallel containers ==="
for i in $(seq 0 $((N_CONTAINERS - 1))); do
  START=$((i * SIMS_PER_CONTAINER))
  NAME="sfincs-rho${RHO}-b${i}"

  echo "  [$i] sims ${START}–$((START + SIMS_PER_CONTAINER - 1)) → $NAME"

  az container create \
    --resource-group "$ACI_RG" \
    --name "$NAME" \
    --image "$IMAGE" \
    --restart-policy Never \
    --os-type Linux \
    --cpu 4 \
    --memory 8 \
    --environment-variables \
        START_IDX="$START" \
        N_SIMS="$SIMS_PER_CONTAINER" \
        RHO="$RHO" \
    --azure-file-volume-account-name "$STORAGE_ACCOUNT" \
    --azure-file-volume-account-key  "$STORAGE_KEY" \
    --azure-file-volume-share-name   "$FILE_SHARE" \
    --azure-file-volume-mount-path   /results \
    --no-wait
done

echo ""
echo "================================================================"
echo " All containers launched."
echo " Results → Storage Explorer: hydratools / dockerdata /"
echo "           summary_corr_05_0000.csv … summary_corr_05_0090.csv"
echo "================================================================"
echo ""
echo "Monitor any container:"
echo "  az container logs -g $ACI_RG -n sfincs-rho${RHO}-b0 --follow"
echo ""
echo "Check all status:"
echo "  az container list -g $ACI_RG -o table"
echo ""
echo "When done, delete containers (storage stays):"
echo "  az group delete --name $ACI_RG --yes --no-wait"
