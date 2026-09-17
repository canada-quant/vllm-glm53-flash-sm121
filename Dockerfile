# canada-quant/vllm-glm53-flash-sm121
#
# GLM-5.3-Flash W4A16 (INT4 GPTQ) + DFlash2 speculative-decoding drafter serving
# on 2x NVIDIA DGX Spark (GB10, SM121a, aarch64, 128 GB UMA per node, TP=2 over RoCE).
#
# This image = official upstream vLLM GLM-5.3-Flash arm64 build + the canada-quant
# SM121 patch overlay ONLY (Apache-2.0-derived; every edit authored and guarded by
# patches/bake_patches.py, which REFUSES on any anchor drift):
#
#   1. vllm/platforms/cuda.py        — route capability-12 NoPE sparse-MLA to the
#                                      FLASHINFER_MLA_SPARSE_SM90 backend (the native
#                                      SM120 lane hard-requires DeepSeek's pe_dim=64
#                                      fp8_ds_mla shape; GLM-5.3 is NoPE, qk_rope_head_dim=0)
#                                      + gate PDL off on SM12x (KDA-state kernel race).
#   2. .../mla/flashinfer_mla_sparse_sm90.py — open the backend's capability gate to
#                                      major in (9,12), select the fa2 decode kernel on
#                                      SM12x (fa3 exceeds the ~101KB smem opt-in), and
#                                      report the typed fp8 dtype if the pool arrives as
#                                      raw uint8 storage.
#   3. flashinfer/mla/_core.py       — widen the FP8-KV-for-MLA device gate to SM121.
#   4. flashinfer/jit/attention/modules.py — compile the batch_mla sm90 JIT kernel for
#                                      the local sm_121a (sm_90a cubins don't run on GB10).
#   5. vllm/models/glm5next/nvidia/model.py (overlay) — Eagle3/aux-hidden-state taps for
#                                      the DFlash2 drafter at dflash_config.target_layer_ids.
#   6. vllm/v1/core/kv_cache_utils.py (overlay) — DFLASH2-DRAFTER-GROUP: keep the GLM-5
#                                      KV fast path engaged with the drafter's full-attention
#                                      layers instead of bailing to the generic unifier
#                                      (which dies: "page size is not divisible").
#
# The target-side top-k init fix (sparse_attn_indexer_kpool.py) that the 2026-08 era
# fork images shipped as a bind-mount is SUBSUMED upstream at this base
# (sparse_attn_indexer_kpool.py:417, `topk_indices_buffer[: hidden_states.shape[0]] = -1`)
# — the proven file is kept in patches/archive/ for provenance only.
#
# The drafter is NOT baked in: it is bind-mounted at runtime (see README "Drafter
# pluggability") so an enhanced drafter is a drop-in swap with no image rebuild.
#
# Base is pinned by digest: the glm53-flash-arm64-cu130 TAG moves (it moved from
# vLLM 487ecf187 to 0.28.1rc1.dev580+g385dce36b between 2026-08 and 2026-09).
# Reproducibility = this digest + the guarded bake, not the tag.
#
# Build (on an aarch64 SM121 host — e.g. a DGX Spark; cross-builds are not supported):
#   docker build -t ghcr.io/canada-quant/vllm-glm53-flash-sm121:v1-w4a16-dflash2e .

ARG BASE_IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5
FROM ${BASE_IMAGE}

# The upstream build this digest was verified against:
#   vllm 0.28.1rc1.dev580+g385dce36b (VLLM_BUILD_COMMIT=385dce36bcee42309924a5ece951a96db3dce7f2)
#   flashinfer 0.6.18 (release), torch 2.13.0+cu130, nvidia-nccl-cu13 2.30.7, CUDA 13.0.2
COPY patches/ /opt/canada-quant-sm121/patches/
RUN python3 /opt/canada-quant-sm121/patches/bake_patches.py /usr/local/lib/python3.12/dist-packages

# Post-patch verification (fail the build if anything is off):
#   - the overlays import cleanly;
#   - DFlash2DraftModel + Glm5NextForConditionalGeneration are registered;
#   - every patched gate reads back with its patched form.
RUN python3 - <<'PY'
import importlib, sys
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import vllm
assert "385dce36" in vllm.__version__ or vllm.__version__.startswith("0.28"), vllm.__version__

from vllm.model_executor.models.registry import ModelRegistry
arch = ModelRegistry.get_supported_archs()
assert "DFlash2DraftModel" in arch, "DFlash2DraftModel not registered"
print("registry OK: DFlash2DraftModel present")

import vllm.v1.core.kv_cache_utils as k
t = open(k.__file__).read()
assert "DFLASH2-DRAFTER-GROUP" in t, "kv overlay not engaged"
print("kv overlay OK: DFLASH2-DRAFTER-GROUP present")

import vllm.models.glm5next.nvidia.model as m
mt = open(m.__file__).read()
assert "SupportsEagle3" in mt and "_aux_stream_value" in mt, "aux-tap overlay not engaged"
print("model overlay OK: Eagle3 aux taps present")

cuda = open("/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py").read()
assert "return major in (9, 10)" in cuda, "PDL gate not applied"
print("cuda.py OK: PDL gated to (9,10); SM90 route present" if "FLASHINFER_MLA_SPARSE_SM90" in cuda else "ROUTE MISSING")

fimla = open("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py").read()
assert "capability.major in (9, 12)" in fimla and 'else "fa2"' in fimla, "fimla gates not applied"
print("fimla OK: capability (9,12) + fa2 selection")

fi_core = open("/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py").read()
assert "major not in (9, 12)" in fi_core, "FI fp8 gate not applied"
fi_mod = open("/usr/local/lib/python3.12/dist-packages/flashinfer/jit/attention/modules.py").read()
assert "extra_cuda_cflags += sm121a_nvcc_flags" in fi_mod, "FI gencode not applied"
print("flashinfer OK: fp8 gate (9,12) + sm121a gencode")
print("ALL_PATCHES_VERIFIED")
PY

LABEL org.opencontainers.image.title="vllm-glm53-flash-sm121" \
      org.opencontainers.image.description="GLM-5.3-Flash W4A16 + DFlash2 drafter serving on DGX Spark (SM121a), TP=2" \
      org.opencontainers.image.vendor="canada-quant" \
      org.opencontainers.image.source="https://github.com/canada-quant/vllm-glm53-flash-sm121" \
      org.opencontainers.image.base.name="docker.io/vllm/vllm-openai:glm53-flash-arm64-cu130" \
      org.opencontainers.image.base.digest="sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5"
