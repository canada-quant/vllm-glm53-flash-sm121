#!/usr/bin/env bash
# launch_glm53_w4a16_dflash2_sm121.sh — GLM-5.3-Flash W4A16 + DFlash2 drafter,
# TP=2 across two DGX Spark GB10 (SM121a) nodes, from the canada-quant SM121 image.
#
# Adapted from the canada-quant serving launcher (sha256 a21f06f4f71d…, published at
# https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-E/blob/main/launch_dflash2_tp2.sh)
# with ONE delta: IMAGE = the canada-quant SM121 image (patches baked in — no
# bind-mounts needed). Engine args are the banked, owner-ruled g4 graphs-ON production
# config (see README "Serving configuration" for provenance).
#
# Usage: launch_glm53_w4a16_dflash2_sm121.sh <0|1>
#   Run the WORKER (1) FIRST, wait ~25 s, then the HEAD (0) on the other node.
#
# Knobs (defaults = the proven 262K g4 config):
#   IMAGE            image to serve from (default ghcr.io/canada-quant/vllm-glm53-flash-sm121:v1-w4a16-dflash2e)
#   MODEL_DIR        W4A16 target weights on THIS node (default /home/pcozz/models/glm-5.3-w4a16-mtp —
#                    adjust to wherever your canada-quant/GLM-5.3-Flash-W4A16-MTP checkout lives)
#   DRAFTER_HOST_PATH  drafter weights dir (default /models/GLM-5.3-Flash-DFlash2-E —
#                    see README "Drafter pluggability" to swap in an enhanced drafter)
#   MAX_MODEL_LEN    262144 (default) | 1048576 (1M; needs KV_CACHE_MEM raise, see README)
#   KV_CACHE_MEM     bytes; default 8053063680 = 8 GiB -> 366,749-token pool @262K (g4)
#   GMU              0.795 (banked g4 value; 0.90 for the 1M serve)
#   GRAPHS           1 (default) = CUDA graphs FULL_AND_PIECEWISE [1,2,4,8,16,24,32];
#                    GRAPHS=0 EAGER=1 = eager fallback
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/canada-quant/vllm-glm53-flash-sm121:v1-w4a16-dflash2e}"
NAME="vllm_node"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
KV_CACHE_MEM="${KV_CACHE_MEM:-8053063680}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
MODEL_DIR="${MODEL_DIR:-/home/pcozz/models/glm-5.3-w4a16-mtp}"
MODEL_PATH="$MODEL_DIR/"DRAFTER_HOST_PATH="${DRAFTER_HOST_PATH:-/models/GLM-5.3-Flash-DFlash2-E}"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-7}"
EAGER="${EAGER:-0}"
GMU="${GMU:-0.795}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
# Your fabric: adjust to your own RoCE/IB interface + subnet (defaults = the
# authors' VLAN 102 point-to-point fabric).
HEAD_IP="${HEAD_IP:-192.168.102.1}"
MPORT="${MPORT:-29521}"
PORT="${PORT:-8000}"
IB_HCA="${IB_HCA:-rocep1s0f1}"
SOCK_IF="${SOCK_IF:-enp1s0f1np1}"
IB_RANGE="${IB_RANGE:-192.168.102.0/24}"

if [ "$EAGER" = "1" ]; then EAGER_FLAG="--enforce-eager"; else EAGER_FLAG=""; fi
GRAPH_ARGS=()
if [ "${GRAPHS:-1}" = "1" ]; then
  GRAPH_ARGS=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}')
fi

NODE_RANK="${1:?usage: launch_glm53_w4a16_dflash2_sm121.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

case "$NODE_RANK" in
  0) HOST_IP="${HOST_IP_RANK0:-192.168.102.1}"; HEADLESS="" ;;
  1) HOST_IP="${HOST_IP_RANK1:-192.168.102.2}"; HEADLESS="--headless" ;;
esac

test -f "$MODEL_DIR/config.json" || { echo "MODEL_DIR/config.json missing: $MODEL_DIR" >&2; exit 1; }
test -f "$DRAFTER_HOST_PATH/config.json" || { echo "DRAFTER_HOST_PATH/config.json missing: $DRAFTER_HOST_PATH" >&2; exit 1; }
test -f "$DRAFTER_HOST_PATH/mask_embedding.pt" || { echo "mask_embedding.pt missing next to the drafter weights — DO NOT SERVE (see README)" >&2; exit 1; }
mkdir -p "$CACHE_HOST_PATH"

# Wedge-safe replace: graceful stop first — NEVER `docker rm -f` a GPU-active
# container on GB10 (UVM wedge). If you are replacing a running serve, use
# `docker stop -t 30 vllm_node` BEFORE running this.
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "refusing to launch over a running $NAME — stop it first (docker stop -t 30 $NAME)" >&2
  exit 1
fi
docker rm "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --memory=118g \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_DIR:$MODEL_PATH" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v "$CACHE_HOST_PATH/flashinfer:/root/.cache/flashinfer" \
  -v "$CACHE_HOST_PATH/tilelang:/root/.tilelang" \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}" \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=$IB_RANGE \
  -e NCCL_SOCKET_IFNAME=$SOCK_IF -e GLOO_SOCKET_IFNAME=$SOCK_IF \
  -e TP_SOCKET_IFNAME=$SOCK_IF -e MN_IF_NAME=$SOCK_IF \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e NCCL_BLOCKING_WAIT=0 \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
  -e TORCH_NCCL_DISABLE_WATCHDOG=1 \
  -v "$DRAFTER_HOST_PATH:/models/dflash2-draft:ro" \
  "$IMAGE" \
  "$MODEL_PATH" \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --gpu-memory-utilization "$GMU" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" --block-size "$BLOCK_SIZE" \
  --speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":'"$SPEC_NUM_TOKENS"'}' \
  --kv-cache-dtype fp8_e4m3 --kv-cache-memory "$KV_CACHE_MEM" \
  $EAGER_FLAG \
  "${GRAPH_ARGS[@]}" \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$HEAD_IP" --master-port "$MPORT" \
  $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP image=$IMAGE"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || true
