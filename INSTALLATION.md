## STEP 1: Create new environment
python -m venv ~/env/cu126 && source ~/env/cu126/bin/activate
python -m pip install -U pip

## STEP 2: Install torch
pip install --index-url https://download.pytorch.org/whl/cu126 \
            torch==2.6.0+cu126 \
            torchvision==0.21.0+cu126 \
            torchaudio==2.6.0+cu126

## STEP 3: Install other libraries
pip install --extra-index-url https://pypi.org/simple \
            ninja packaging setuptools>=68 setuptools_scm>=8 cmake>=3.27 \
            accelerate==0.30.1 transformers==0.20.0 peft==0.12.0 seqeval==1.2.2 hydra-core==1.3.2 \
            vllm==0.8.5.post1

## STEP 4: Install unsloth
pip install --no-cache-dir --no-deps \
  "unsloth[cu126-ampere-torch260] @ git+https://github.com/unslothai/unsloth.git"

OR

pip install --no-cache-dir --no-deps \
  "unsloth[cu126-ampere-torch260] @ git+https://github.com/unslothai/unsloth.git@main" \
  "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git@main"

## STEP 5: Build flash-attn
export TORCH_CUDA_ARCH_LIST=8.0   # only A100 kernels → 3-5 min
export MAX_JOBS=$(nproc)

pip install flash-attn --no-build-isolation --extra-index-url https://pypi.org/simple


## Verify Installation
python - <<'PY'
import importlib.metadata as md, torch
for p in ("torch","vllm","flash_attn","unsloth",
          "accelerate","peft","seqeval","hydra-core"):
    print(f"{p:12s}", md.version(p))
print("\nCUDA runtime in Torch:", torch.version.cuda)
PY