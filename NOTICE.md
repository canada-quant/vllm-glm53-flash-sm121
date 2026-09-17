# NOTICE — attribution & license chain

## This repository (canada-quant/vllm-glm53-flash-sm121)

Authored content (Dockerfile, `patches/bake_patches.py`, overlays, launcher, docs):
**Apache-2.0**, © 2026 canada-quant.

## Upstream artifacts this image builds on

| Artifact | License | Role |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) fork build `0.1.dev20051+g487ecf187` (re-hosted at `ghcr.io/canada-quant/vllm-glm53-flash-base:sm121-v11-dflash2`, digest `sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`) | Apache-2.0 | v2 base image: GLM-5.3-Flash model support, DFlash2 spec-decode, speculators-format drafter loading, the mask-embedding loader, the SM121 sparse-MLA routing |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 0.6.18.dev20260819 | Apache-2.0 | Sparse-MLA attention kernels (fa2 path) |
| [DFlash2](https://github.com/z-lab/dflash) ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) | (method paper) | The speculative-decoding method |

### v1 experimental base (not the default build)

| Artifact | License | Role |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) @ `385dce36bcee42309924a5ece951a96db3dce7f2` (image `vllm/vllm-openai:glm53-flash-arm64-cu130`, digest `sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5`) | Apache-2.0 | `Dockerfile.experimental-upstream` base — boots through load, dies at sparse-MLA warmup on SM121 (see README "Known failures") |

The baked patch files under `patches/fork/` are derived from the corresponding
vLLM Apache-2.0 files (fork era) and remain Apache-2.0; the modifications (the
top-k init fix; DFLASH2-DRAFTER-GROUP KV fast-path retention) are
canada-quant's, developed for the GLM-5.3-Flash W4A16 dual-DGX-Spark serving
stack. The kvcu hunk was previously published (H200 lane) inside
`canada-quant/dgx-spark-sm121@77ace0b3`.

## Friendly debt (no code copied)

A public community GLM-5.3-Flash DGX-Spark bring-up repository documented the SM121 day-0 pitfalls (the SM90-NoPE backend being the only
capability-12 route for NoPE sparse MLA, the fa2 smem constraint, the PDL race) and
the official-image-plus-Dockerfile build shape. **Their repository carries no
license file**, so nothing from it is copied into this repo or baked into the
image — every patch here is canada-quant-authored against upstream bytes (their
routing insight is acknowledged as prior art). The historical fork image
(`radixark/vllm-glm53-flash:sm121-v11-dflash2`, now 404 on Docker Hub; re-hosted at
`ghcr.io/canada-quant/vllm-glm53-flash-base:sm121-v11-dflash2`) that carried those
same-class fixes is **not** a layer of this image.

## Historical fork files

`patches/archive/sparse_attn_indexer_kpool.py` (sha256
`8a3ecfb0bab2441dd7417ed00a10d142191496149f88e5fe79fcfaea4b160980`, 46,355 bytes)
is a byte-copy of the fork-era top-k fix file (`patches/fork/` carries the
serving copy; the archive entry predates the bake and is kept for the 2026-08-28
provenance chain). Derived from vLLM's Apache-2.0
`sparse_attn_indexer_kpool.py` of the fork era.

The historical fork image (`radixark/vllm-glm53-flash:sm121-v11-dflash2`, now
404 on Docker Hub; re-hosted byte-identically at
`ghcr.io/canada-quant/vllm-glm53-flash-base:sm121-v11-dflash2`, digest
`sha256:4def0ef6…`) **is a layer of the v2 image** — as the pinned base.
