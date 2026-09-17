# vllm-glm53-flash-sm121

**One-command serving of [GLM-5.3-Flash W4A16](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) + a [DFlash2 speculative-decoding drafter](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-E) on 2× NVIDIA DGX Spark (GB10, SM121a), tensor-parallel over RoCE.**

- **Image**: `ghcr.io/canada-quant/vllm-glm53-flash-sm121:v2-w4a16-dflash2e` (aarch64)
- **Lineage**: the public DGX-Spark bring-up image `ghcr.io/canada-quant/vllm-glm53-flash-base:sm121-v11-dflash2` (pinned digest `sha256:4def0ef6…`; vLLM fork `0.1.dev20051+g487ecf187`, FlashInfer `0.6.18.dev20260819`, CUDA 13.0) **+ the two serving-critical canada-quant patches baked in** — `sparse_attn_indexer_kpool.py` (NoPE sparse-indexer top-k fix) and `kv_cache_utils.py` (`DFLASH2-DRAFTER-GROUP`), both sha256-gated at build time to the exact production bytes. No host-side patch bind-mounts needed to serve. (An experimental upstream-nightly-based build exists at `Dockerfile.experimental-upstream`; it is known-broken on SM121 — see "Known failures".)

## TL;DR — two nodes, four commands

```bash
# 1. On BOTH nodes: pull the image, fetch the weights
docker pull ghcr.io/canada-quant/vllm-glm53-flash-sm121:v2-w4a16-dflash2e
huggingface-cli download canada-quant/GLM-5.3-Flash-W4A16-MTP --local-dir /models/glm53-w4a16
huggingface-cli download canada-quant/GLM-5.3-Flash-DFlash2-E --local-dir /models/GLM-5.3-Flash-DFlash2-E

# 2. Copy the launcher from this repo to both nodes, then:
bash launch_glm53_w4a16_dflash2_sm121.sh 1 &   # WORKER first (rank 1)
sleep 25
bash launch_glm53_w4a16_dflash2_sm121.sh 0     # then HEAD (rank 0)

# 3. Wait for boot (~6–10 min cold; watch for the gates below), then:
curl -s http://<head>:8000/v1/models | head -c 400
```

The launcher defaults are the banked production config: TP=2 + expert-parallel, 262K context, `fp8_e4m3` KV, `num_speculative_tokens=7`, CUDA graphs `FULL_AND_PIECEWISE [1,2,4,8,16,24,32]`, `gpu-memory-utilization 0.795`. Override via the env knobs in the launcher header.

## Prerequisites (hardware + fabric)

| Item | Requirement |
|---|---|
| Nodes | 2× DGX Spark (GB10, SM121a, 128 GB unified memory each) |
| Interconnect | RoCE/IB between the two (the authors use VLAN 102, MTU 9000, `rocep1s0f1`/`enp1s0f1np1`) — adjust `IB_HCA`/`SOCK_IF`/`IB_RANGE` in the launcher |
| Disk | ~200 GiB per node for weights (~178 GiB target + ~6.2 GB drafter) + image (~21 GB) |
| Weights | `canada-quant/GLM-5.3-Flash-W4A16-MTP` (target) + `canada-quant/GLM-5.3-Flash-DFlash2-E` (drafter) |
| Model card | Read the target's card for the quant design (INT4 g128 on routed experts only) and the full hardware matrix |

## Build the image yourself

Only if you want to modify the patches. The published image was built on an aarch64 SM121 host:

```bash
docker build -t ghcr.io/canada-quant/vllm-glm53-flash-sm121:v2-w4a16-dflash2e .
```

The build applies `patches/bake_patches.py` to the pinned base and **fails** (anchor refused) if upstream drifted. `patches/fork/` carries the two serving-critical patches baked into v2 — byte-identical to the production stack's runtime-mounts, sha256-gated by the Dockerfile's RUN gate. `patches/overlays/` holds the full-file overlays used only by `Dockerfile.experimental-upstream` (the known-broken upstream-nightly arm — see "Known failures"). `patches/archive/` retains the historical fork-era `sparse_attn_indexer_kpool.py` (sha256 `8a3ecfb0bab2…`) for provenance — identical bytes to the fork copy it documents.

## Serving configuration (provenance)

The launcher's engine args are byte-derived from the authors' banked production configuration (owner-ruled "g4", 2026-09-16): TP=2, EP on, `--block-size 2304` (engine raises to 4608 for mamba-page alignment), `GMU 0.795`, `--kv-cache-memory 8053063680` (8 GiB → pool 366,749 tokens @262K), `fp8_e4m3` KV, K=7, graphs ON. Expected boot gates (verify in `docker logs vllm_node`):

1. `Using HND KV cache layout for FLASHINFER_MLA_SPARSE_SM90 backend`
2. `DFLASH2-DRAFTER-GROUP: GLM-5-Next fast path engaged with a spec-decode drafter group (drafter_layers=8, mamba_groups=4, tail_group=True)`
3. `GPU KV cache size: 366,749 tokens` (the 8-GiB discriminator)
4. `Loaded DFlash mask embedding for mask_token_id 154856 from mask_embedding.pt` — **absence means the mask was silently ignored; do not serve**
5. `FULL_AND_PIECEWISE` capture `[1,2,4,8,16,24,32]` in the engine config dump

### 1M context

`MAX_MODEL_LEN=1048576 GMU=0.90 KV_CACHE_MEM=9663676416` (the launcher's published 1M guidance; pool ~1.36M tokens ≈ 1.30× a full 1M request).

### Hard constraints (measured, not stylistic)

- `num_speculative_tokens` **must be 7** (= `block_size − 1`). Other counts boot-wedge the DFlash2 stack.
- `mask_embedding.pt` must sit next to the drafter weights (the launcher hard-checks it).
- Cold boot ≈ 6–10 min (`VLLM_ENGINE_READY_TIMEOUT_S=3600` is load-bearing — cold JIT otherwise kills boot).
- Graceful `docker stop -t 30` only — never `rm -f` a GPU-active container on GB10 (UVM wedge).

## Drafter pluggability (versioning protocol)

The drafter is **not** baked into the image. It is a bind-mounted directory (`DRAFTER_HOST_PATH`), and the image serves any GLM-5.3-Flash-compatible DFlash2 drafter without a rebuild:

```bash
# Swap to an enhanced drafter (example):
DRAFTER_HOST_PATH=/models/GLM-5.3-Flash-DFlash2-F bash launch_glm53_w4a16_dflash2_sm121.sh 1
```

**Rules for a drop-in drafter:**

1. **Dir layout**: `model.safetensors` + `config.json` + `mask_embedding.pt` (+ optional `PROVENANCE.txt`), exactly like [`GLM-5.3-Flash-DFlash2-E`](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-E).
2. **K follows the architecture**: `num_speculative_tokens = block_size − 1` (E: block 8 → K=7). Set `SPEC_NUM_TOKENS` to match; anything else boot-wedges.
3. **Version discipline**: drafters are versioned by their HF repo id (`…-E`, `…-F`, …) with the config's `dflash_config.target_layer_ids` as the compatibility contract — the target-side aux-tap overlay honors whatever the drafter's config declares (9 taps for E: `[5,9,14,19,24,28,33,38,42]`). A drafter that changes tap geometry needs no image change; it needs its config to be truthful.
4. **Verify after swap**: boot gate 4 (mask-loader line) + a sanity generation (below). No image rebuild, no target-side change.

## Sanity generation

```bash
curl -s http://<head>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "In one sentence: why does speculative decoding help here?"}],
  "max_tokens": 64
}' | python3 -m json.tool | head -20
```

## Repository layout

```
Dockerfile                              pinned base + guarded bake + post-build verification
patches/bake_patches.py                 all in-place edits (guarded, idempotent, refuses drift)
patches/overlays/glm5next_nvidia_model.py   aux-tap overlay (sha256 ac9c4f26…)
patches/overlays/kv_cache_utils.py      DFLASH2-DRAFTER-GROUP overlay (sha256 0094aad4…)
patches/archive/sparse_attn_indexer_kpool.py  fork-era top-k fix (8a3ecfb0…, provenance)
scripts/launch_glm53_w4a16_dflash2_sm121.sh  the serving launcher
NOTICE.md                               attribution + license chain
```

## License & attribution

Apache-2.0 (this repo's authored content). The v2 image is built on the
`canada-quant/vllm-glm53-flash-base` public DGX-Spark bring-up image (itself an
Apache-2.0 vLLM fork build) + the two canada-quant patches in `patches/fork/`
(sha-gated at build). The v1 experimental image is built on official upstream
`vllm/vllm-openai` Apache-2.0 artifacts. See [`NOTICE.md`](NOTICE.md) — which
also documents the friendly-debt line to the community bring-up author's published routing insight.

## Known failures (do not repeat)

- **v1 / upstream-nightly base (`Dockerfile.experimental-upstream`)**: boots
  through target + drafter load, then dies at FlashInfer sparse-MLA warmup on
  real SM121 hardware — `flashinfer_mla_sparse_sm90.py:483 forward_mqa →
  Failed to run MLA, error: invalid argument` — with AND without
  `--no-enable-flashinfer-autotune` (gate attempts 2026-09-17 05:09Z / 05:29Z).
  The two `patches/fork/` bytes are load-bearing for this path and have no
  upstream-nightly equivalent yet. Rebase stays parked until that lands.


## Status

Serving-validated by the authors on 2× DGX Spark. This image is the container their production pair runs.
