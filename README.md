# hip-moe — building a CUDA-only MoE expert-cache fork on ROCm, with numbers

`neurall/llama.cpp` ("RocketLLama", branch `release`) keeps every MoE expert in system RAM and turns each
GPU's free VRAM into a **cache** of the experts tokens actually use: the GPUs compute cached experts while
the CPU computes the misses in parallel. Upstream llama.cpp instead pins whole layers on the GPU.

The fork is written against the CUDA backend. This repo is what it takes to build and run it on **AMD
hardware under ROCm**, measured, plus the result nobody expected.

No NVIDIA card was involved in anything here.

## The port: three NVIDIA-isms

`ggml/src/ggml-cuda/ggml-cuda.cu` does not compile with `-DGGML_HIP=ON` as shipped. Each fault is small,
each has a two-line fix, and all three are in `patches/0001-hip-port.patch`.

| # | fault | fix |
|---|---|---|
| 1 | `#include <nvtx3/nvToolsExt.h>` unguarded on Linux — NVTX is an NVIDIA SDK | `#if __has_include(...)`, else define the two range macros to no-ops |
| 2 | the expert-observation probe uses `cudaStreamCaptureStatus`, `...StatusNone`, `cudaStreamIsCapturing` — HIP has no aliases for these names | `#define` them to the `hip*` equivalents under `GGML_USE_HIP && __HIP_PLATFORM_AMD__` |
| 3 | the zero-copy staging kernels use `__ldcv` (no HIP equivalent) and `cudaHostRegisterMapped` (exists as `hipHostRegisterMapped`) | 16-byte path → four `__builtin_nontemporal_load`s; 8-bit path → one; flag aliased |

**The one non-obvious part, so nobody re-derives it:** `*reinterpret_cast<const volatile T *>(p)` is the
usual replacement for `__ldcv` and it works for scalars — but it cannot construct a `HIP_vector_type`, and
`__builtin_nontemporal_load` rejects HIP's vector types outright (*"address argument to nontemporal builtin
must be a pointer to integer, float, pointer, or a vector of such types"*). Four scalar non-temporal loads
is the shape that compiles and keeps the intent: do not serve a stale line of a reused pinned block.

Build dependencies a runtime-only ROCm install does not carry: **`rocblas` and `hipblas`**.

```sh
cmake -B build -G Ninja -DGGML_HIP=ON -DGPU_TARGETS="gfx1201" -DCMAKE_BUILD_TYPE=Release
ninja -C build llama-cli
```

## The result, on one card

Model: `Qwen3-30B-A3B-Q4_K_M.gguf`, **18.5 GB** — larger than the card's 16.2 GiB, so the cache has to
engage for the model to run at all. Card: RX 9070 XT (gfx1201, RDNA4). Prompt: one sentence; 96 tokens out;
`-c 4096`; single turn.

Both arms carry **identical placement** (`-ngl 99 --cpu-moe -nr`). The only difference is the cache.

| arm | flags | prompt t/s | decode t/s |
|---|---|---|---|
| stock | `--fork off -ngl 99 --cpu-moe -nr` | 99.1 | **37.6** |
| expert cache | `-ngl 99 --cpu-moe --moe-expert-cache -1 -nr` | 100.5 | **65.8** |

**Decode +75%. Prompt flat**, which is the expected shape: the fork's prompt gains are long-prompt and
multi-GPU, and a one-sentence prompt gives the cache nothing to amortize.

## The result nobody expected: three cards lose to one

Built a second tree with `-DGPU_TARGETS="gfx900;gfx1201"` (the other two cards are gfx900 — ROCm 7.2 still
compiles for them, outside official support). Same model, same flags, only the visible device set differing.

| devices | prompt t/s | decode t/s |
|---|---|---|
| 9070 XT alone (multi-arch build) | **189.6** | **77.2** |
| 9070 XT alone (gfx1201-only build) | 100.5 | 65.8 |
| all three (9070 XT + 2× gfx900) | 74.4 | 43.5 |

**Adding two more cards, each with the same 16 GiB of VRAM, costs ~44 % of decode and ~61 % of prompt.**
Reproduced three times, cold-state and with the learned hot-expert map warm.

The null arm matters: the single-card figure is the **same multi-arch build**, so this is not a build
artefact. The likely cause is in the fork's own documentation — the scheduler synchronises at every layer,
so a layer owned by a card on a slower link makes the fast card wait. The fork's reference rig is two
*matched* 3090s, which is the case where a split pays. An unmatched trio, where the fastest card waits on
two much slower ones, is where it does not.

Stated as a hypothesis and not a finding: separating link cost from compute cost needs `--moe split=` and
`-ts` arms, which this repo does not yet have.

## Reproduce everything

```sh
./repro.sh /path/to/moe-model.gguf
```

## What this is not

Not a fork of the fork. One patch, applied to `release` at commit `27cc499`. All credit for the expert
cache is the fork author's; the AMD port and the measurements are the work.
