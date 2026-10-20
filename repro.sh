#!/bin/sh
# Build neurall/llama.cpp (RocketLLama) for ROCm, apply the HIP port, and run both arms.
# usage: ./repro.sh /path/to/moe-model.gguf [gfx-target]
set -eu
MODEL="${1:?usage: repro.sh /path/to/moe-model.gguf [gfx1201]}"
TARGET="${2:-gfx1201}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HOME/rocketllama-repro}"

command -v hipcc >/dev/null || { echo "no hipcc on PATH — install a ROCm toolchain first"; exit 1; }
command -v cmake >/dev/null || { echo "no cmake"; exit 1; }
ls /opt/rocm/lib/cmake/hipblas* >/dev/null 2>&1 || echo "warning: hipblas not found — install rocblas and hipblas"

[ -d "$WORK/.git" ] || git clone --depth 1 -b release https://github.com/neurall/llama.cpp.git "$WORK"
cd "$WORK"
git checkout -- . 2>/dev/null || true
git apply --check "$HERE/patches/0001-hip-port.patch" && git apply "$HERE/patches/0001-hip-port.patch"

cmake -B build -G Ninja -DGGML_HIP=ON -DGPU_TARGETS="$TARGET" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF >/dev/null
ninja -C build llama-cli

P="Write one paragraph explaining why a cache that misses is not the same as a cache that lies."

echo
echo "=== arm 1: stock placement, no expert cache ==="
./build/bin/llama-cli -m "$MODEL" --fork off -ngl 99 --cpu-moe -nr -c 4096 -n 96 -p "$P" --no-warmup --single-turn 2>&1 | grep -E "Prompt:"

echo
echo "=== arm 2: same placement, expert cache on ==="
./build/bin/llama-cli -m "$MODEL" -ngl 99 --cpu-moe --moe-expert-cache -1 -nr -c 4096 -n 96 -p "$P" --no-warmup --single-turn 2>&1 | grep -E "Prompt:"
