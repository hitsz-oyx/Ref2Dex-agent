# Source from the repository root before ObjectForesight model stages.
# Downloads run separately online; inference uses acquired local model files.
export REF5_OUTPUT_ROOT="$PWD/outputs/cm-pointflow-effect-pretrain/ref5-data-expansion-20261007"
export TMPDIR="$PWD/tmp/ref5-data-expansion"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TORCH_EXTENSIONS_DIR="$REF5_OUTPUT_ROOT/torch-extensions"
export HF_HOME="$REF5_OUTPUT_ROOT/hf-cache"
export XDG_CACHE_HOME="$REF5_OUTPUT_ROOT/cache"
export NUMBA_CACHE_DIR="$REF5_OUTPUT_ROOT/numba-cache"
export WARP_CACHE_PATH="$REF5_OUTPUT_ROOT/warp-cache"
export U2NET_HOME="$REF5_OUTPUT_ROOT/u2net-cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPARSE_ATTN_BACKEND=flash_attn
