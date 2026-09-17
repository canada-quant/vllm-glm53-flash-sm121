#!/usr/bin/env python3
"""bake_patches.py — apply the canada-quant SM121 (DGX Spark GB10) vLLM patch overlay.

Every edit is GUARDED: the expected upstream anchor must appear exactly once, or this
script REFUSES (exit 1) — so if a future base image drifts, the build fails loudly
instead of silently producing a broken image. Idempotent: already-applied edits are
skipped. All edits are authored by canada-quant on top of Apache-2.0 upstream vLLM /
FlashInfer files; see NOTICE.md for the full attribution chain.

Usage: python3 bake_patches.py <dist-packages root>
       (default /usr/local/lib/python3.12/dist-packages)
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages")
PATCH_DIR = Path(__file__).resolve().parent
VLLM = ROOT / "vllm"
FI = ROOT / "flashinfer"

applied: list[str] = []


def edit(path: Path, old: str, new: str, name: str) -> None:
    t = path.read_text()
    if new in t:
        print(f"SKIP  {name}: already applied")
        return
    c = t.count(old)
    if c != 1:
        sys.exit(f"REFUSE {name}: anchor found {c}x (expected 1) in {path} — base drifted; re-verify")
    path.write_text(t.replace(old, new))
    applied.append(name)
    print(f"OK    {name}")


# ---------------------------------------------------------------- 1. cuda.py --
CUDA = VLLM / "platforms" / "cuda.py"

# 1a. Route capability-12 to the SM90 NoPE sparse-MLA backend (proven v11 form:
#     add the backend to the major==12 candidate list). GLM-5.3-Flash is NoPE
#     (qk_rope_head_dim=0); the native SM120 lane hard-requires DeepSeek's
#     pe_dim==64 fp8_ds_mla shape and dies at KV-cache init.
edit(
    CUDA,
    """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]
""",
    """        elif device_capability.major == 12:
            # canada-quant patch (SM121, 2026-09): GLM-5.3-Flash NoPE sparse MLA
            # (qk_rope_head_dim=0) needs the SM90 NoPE backend; the SM120 lane
            # hard-requires pe_dim==64 fp8_ds_mla. Proven on DGX Spark GB10.
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]
""",
    "cuda.py: SM90 NoPE route on capability-12",
)

# 1b. Gate PDL off on SM12x: Triton KDA-state kernels race under PDL on GB10
#     (boots NaN or not depending on launch timing).
edit(
    CUDA,
    """        except Exception:
            return False
        return major >= 9
""",
    """        except Exception:
            return False
        # canada-quant patch (SM121, 2026-09): PDL lowering is unvalidated on
        # SM12x (GB10) and races on KDA state kernels there; keep it to
        # Hopper/Blackwell-datacenter.
        return major in (9, 10)
""",
    "cuda.py: PDL gated to (9, 10)",
)

# --------------------------------------------- 2. flashinfer_mla_sparse_sm90 --
FIMLA = VLLM / "v1" / "attention" / "backends" / "mla" / "flashinfer_mla_sparse_sm90.py"

# 2a. Open the backend's capability gate to capability-12 (proven v11 form).
edit(
    FIMLA,
    "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n"
    "        return capability.major == 9\n",
    "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n"
    "        # canada-quant patch (SM121, 2026-09): the SM90 NoPE sparse-MLA\n"
    "        # backend serves GLM-5.3-Flash on DGX Spark GB10 (SM121a).\n"
    "        return capability.major in (9, 12)\n",
    "fimla: capability gate (9, 12)",
)

# 2b. Select the fa2 decode kernel on SM12x: fa3 exceeds SM121's ~101KB
#     shared-memory opt-in budget ("Failed to run MLA, error: invalid argument").
edit(
    FIMLA,
    '            backend="fa3",\n',
    '            # canada-quant patch (SM121, 2026-09): fa3 exceeds SM121\'s smem\n'
    '            # opt-in budget; fa2 is the proven decode kernel on GB10.\n'
    '            backend=("fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2"),\n',
    "fimla: fa2 on SM12x",
)

# 2c. If the KV pool arrives as raw uint8 storage (some --kv-cache-dtype fp8_e4m3
#     mappings), report the typed dtype to flashinfer's plan(); no-op otherwise.
edit(
    FIMLA,
    "        self.kv_dtype = kv_dtype\n",
    "        # canada-quant patch (SM121, 2026-09): report the typed dtype when the\n"
    "        # pool arrives as raw uint8 storage; flashinfer validates typed dtypes.\n"
    "        self.kv_dtype = torch.float8_e4m3fn if kv_dtype is torch.uint8 else kv_dtype\n",
    "fimla: uint8->typed fp8 shim",
)

# --------------------------------------------------- 3. flashinfer _core.py --
FI_CORE = FI / "mla" / "_core.py"

# 3a. Widen the FP8-KV-for-MLA device gate to SM121 (proven v11 form).
edit(
    FI_CORE,
    "            major, minor = get_compute_capability(self.device)\n"
    "            if major != 9:\n"
    "                raise ValueError(\n"
    '                    "FP8 kv_data_type for MLA requires an SM90 (Hopper) device, "\n'
    '                    f"got SM{major}{minor}."\n'
    "                )\n",
    "            major, minor = get_compute_capability(self.device)\n"
    "            # canada-quant patch (SM121, 2026-09): the SM90-lane sparse-MLA\n"
    "            # kernels serve fp8 KV on SM121 (DGX Spark GB10, proven).\n"
    "            if major not in (9, 12):\n"
    "                raise ValueError(\n"
    '                    "FP8 kv_data_type for MLA requires an SM90 (Hopper) device, "\n'
    '                    f"got SM{major}{minor}."\n'
    "                )\n",
    "FI _core.py: FP8-KV gate (9, 12)",
)

# ------------------------------------------- 4. flashinfer JIT modules.py --
FI_MOD = FI / "jit" / "attention" / "modules.py"

# 4a. Import sm121a flags (jit/core.py:138 defines sm121a_nvcc_flags).
edit(
    FI_MOD,
    "\n    sm90a_nvcc_flags,\n",
    "\n    sm90a_nvcc_flags,\n    sm121a_nvcc_flags,\n",
    "FI modules.py: import sm121a flags",
)

# 4b. Compile the batch_mla sm90 JIT kernel for the LOCAL sm_121a instead of
#     sm_90a (sm_90a cubins don't run on GB10: "no kernel image is available").
edit(
    FI_MOD,
    "        extra_cuda_cflags += sm90a_nvcc_flags\n",
    "        # canada-quant patch (SM121, 2026-09): was sm90a_nvcc_flags — sm_90a\n"
    "        # cubins don't run on sm_121a. The kernel is CUDA-core (no wgmma)\n"
    "        # and compiles for 121a (proven on GB10).\n"
    "        extra_cuda_cflags += sm121a_nvcc_flags\n",
    "FI modules.py: sm121a gencode",
)

# --------------------------------------------------------- 5. the overlays --
# Byte-exact drop-in files (our own authored ports), replacing upstream files
# whose patch surface is too large for in-place edits:
#   glm5next_nvidia_model.py — Eagle3/aux-hidden-state taps for the DFlash2
#       drafter (SupportsEagle3 + aux_hidden_states capture at
#       dflash_config.target_layer_ids; the mHC-contracted stream matches what
#       the drafter was trained on). sha256 ac9c4f26b378… (banked chain).
#   kv_cache_utils.py — DFLASH2-DRAFTER-GROUP: keep the GLM-5 KV fast path
#       engaged with the drafter's full-attention layers (exact-fit slot-share)
#       instead of bailing to the generic unifier ("page size is not divisible"
#       death). sha256 0094aad4c3de… (this repo, ported from the banked H200
#       v3 rev b894ad44… onto the pinned base).
OVERLAYS = {
    PATCH_DIR / "overlays" / "glm5next_nvidia_model.py":
        VLLM / "models" / "glm5next" / "nvidia" / "model.py",
    PATCH_DIR / "overlays" / "kv_cache_utils.py":
        VLLM / "v1" / "core" / "kv_cache_utils.py",
}
for src, dst in OVERLAYS.items():
    if not src.exists():
        sys.exit(f"REFUSE overlay: {src} missing from the build context")
    cur = dst.read_text()
    new = src.read_text()
    if cur == new:
        print(f"SKIP  overlay {dst.name}: already in place")
        continue
    shutil.copy2(src, dst)
    applied.append(f"overlay {dst.name}")
    print(f"OK    overlay {dst.name} -> {dst}")

print(f"\nBAKE_COMPLETE: {len(applied)} edits applied, {applied}")
