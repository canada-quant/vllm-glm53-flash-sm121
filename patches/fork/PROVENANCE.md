# PROVENANCE — patches/fork/ (the two serving-critical fork patches, baked in v2)

These are the exact bytes the production S1+S2 serving stack has run since 2026-09-12
(as runtime bind-mounts over the fork image's files). v2 bakes them at the same
import paths inside the image — byte-identical, sha256-gated at build time.

| file | in-image path | sha256 |
|---|---|---|
| `sparse_attn_indexer_kpool.py` | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | `8a3ecfb0bab2441dd7417ed00a10d142191496149f88e5fe79fcfaea4b160980` |
| `kv_cache_utils.py` | `vllm/v1/core/kv_cache_utils.py` | `b894ad440cd4722342484cc751ba65a480c6903368af168386b86ca6768288e0` |

## sparse_attn_indexer_kpool.py (the kpool top-k init fix)

Target-side fix for the NoPE sparse indexer's top-k initialization on the
GLM-5.3-Flash sparse-MLA path (SM121). Authored in the 2026-08 SM121 bring-up
(sm121 lane); preserved verbatim from `/home/pcozz/patches/sparse_attn_indexer_kpool.py`
(ts 2026-08-28). The upstream-nightly equivalent (`v1` base,
vLLM `0.28.1rc1.dev580+g385dce36b`) reorganized this module around a
`kpool_compress` ops namespace and its stock form **dies at warmup** in the
sparse-MLA path (`flashinfer_mla_sparse_sm90.py:483 forward_mqa →
tvm.error.InternalError: Check failed: (status == cudaSuccess) is false:
Failed to run MLA, error: invalid argument` — observed on real SM121 hardware,
2026-09-17, with and without FlashInfer autotune). That failure is why v2 exists.

## kv_cache_utils.py (DFLASH2-DRAFTER-GROUP, G1 port rev 3)

Port of the H200 DFLASH-V1-DRAFTER hunk onto the fork's
`vllm/v1/core/kv_cache_utils.py`, so the GLM-5 KV fast path accepts a
full-attention (`FullAttentionSpec`) DFlash v1/v2 drafter next to the
sliding-window drafters it already handles. Authored 2026-09-12 (spark lane),
kernel-opt-reviewed through rev 3 (rev 3 adds the `DFLASH2-DRAFTER-GROUP`
success-marker log line — logging-only, zero grouping-semantics delta vs rev 2).
Full review history: `patches/sm121-dflash-v1-drafter/PROVENANCE.md` in the
canada-quant/spark-cluster repo.

## Verification chain (2026-09-17)

- Live serve on these bytes (bind-mount form): S1+S2 rollback boot 06:07:53Z,
  both ranks Up, health 200, g4 fuses green (KV pool 366,749 tokens @ 8 GiB,
  FULL_AND_PIECEWISE capture [1,2,4,8,16,24,32], mask-loader ×1 each rank),
  gateway `:4001` glm-5.3-flash smoke HTTP 200 (probe 06:29:20Z).
- v2 image build asserts both sha256s at build time and refuses otherwise.
