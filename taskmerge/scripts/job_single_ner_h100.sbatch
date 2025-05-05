#!/bin/bash
#SBATCH --job-name=ner_span_h100 # Changed name slightly for clarity
#SBATCH --partition=gpu           # Adjust partition name if needed
#SBATCH --gres=gpu:h100:1       # Request 1 H100 GPU
#SBATCH --time=03:00:00         # Potentially shorter time for H100
#SBATCH --cpus-per-task=16      # Request CPUs (adjust if needed)
#SBATCH --mem=128G            # Potentially more memory for H100 run
#SBATCH --output=logs/ner_span_h100-%j.out # Log file location
#SBATCH --error=logs/ner_span_h100-%j.err  # Error file location

# --- Environment Setup ---
echo "Loading modules..."
module purge
# Adjust module load command based on your HPC environment for H100
# Might be the same or different from A100 modules
module load 2024 cuDNN/9.5.0.50-CUDA-12.6.0 NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
echo "Modules loaded."

# Activate Python environment
ENV_PATH="$HOME/envs/taskmerge/bin/activate" # Example from spec
if [ -f "$ENV_PATH" ]; then
    echo "Activating Python environment: $ENV_PATH"
    source "$ENV_PATH"
else
    echo "Python environment not found at $ENV_PATH. Using default python."
fi

# --- Variables and Paths ---
WORKROOT="$HOME/data/storage_hpc_nishant/taskmerge" # Corrected base path
PERSIST_DATA_ROOT="$WORKROOT/data/union_span/ner"
PERSIST_TOKENIZER_ROOT="$WORKROOT/tokenizers/llama3_biomerge_tok"
SCRATCH_DIR=${TMPDIR:-$WORKROOT/scratch/$SLURM_JOB_ID}
TMP_DATA="$SCRATCH_DIR/data/union_span/ner"
TMP_TOKENIZER="$SCRATCH_DIR/tokenizers/llama3_biomerge_tok"
TMP_OUT="$SCRATCH_DIR/outputs"
ARCHIVE_DIR="$WORKROOT/archive"

echo "Work root: $WORKROOT"
echo "Persistent data: $PERSIST_DATA_ROOT"
echo "Scratch directory: $SCRATCH_DIR"
echo "Temporary data: $TMP_DATA"
echo "Temporary outputs: $TMP_OUT"

# --- Data Preparation ---
echo "Creating directories in scratch space..."
mkdir -p "$TMP_DATA" "$TMP_TOKENIZER" "$TMP_OUT"

echo "Copying data to scratch space..."
rsync -a --info=progress2 "$PERSIST_DATA_ROOT/" "$TMP_DATA/"
echo "Data copied."

echo "Copying tokenizer to scratch space..."
rsync -a --info=progress2 "$PERSIST_TOKENIZER_ROOT/" "$TMP_TOKENIZER/"
echo "Tokenizer copied."

# --- Set W&B Project --- #
export WANDB_PROJECT=${WANDB_PROJECT:-TaskMerge}
export WANDB_DIR=$TMP_OUT
echo "WANDB_PROJECT set to: $WANDB_PROJECT"

# --- Run Training ---
echo "Starting training run..."
cd "$WORKROOT"

SRUN_ARGS=(
    "srun"
    "--cpu-bind=none"
)
TORCHRUN_ARGS=(
    "torchrun"
    "--nproc_per_node=1"
    "--nnodes=1"
    "-m"
    "taskmerge.src.main"
)
HYDRA_ARGS=(
    "task=ner_span"
    "model=unsloth_lora_bf16"
    "runtime=h100" # Use the H100 runtime config
    "task.datamodule.data_dir=$TMP_DATA"
    "model.tokenizer_path=$TMP_TOKENIZER"
    "runtime.training.output_dir=$TMP_OUT"
    # "+task.evaluation.run_test_eval=true"
)

CMD=("${SRUN_ARGS[@]}" "${TORCHRUN_ARGS[@]}" "${HYDRA_ARGS[@]}")
echo "Running command: ${CMD[@]}"

"${CMD[@]}"
EXIT_CODE=$?

echo "Training finished with exit code: $EXIT_CODE"

# --- Archive Results (Optional) ---
if [ $EXIT_CODE -eq 0 ] && [ -d "$ARCHIVE_DIR" ]; then
    echo "Archiving results from $TMP_OUT to $ARCHIVE_DIR..."
    ARCHIVE_PATH="$ARCHIVE_DIR/ner_span_h100_$SLURM_JOB_ID"
    mkdir -p "$ARCHIVE_PATH"
    rsync -a --info=progress2 "$TMP_OUT/" "$ARCHIVE_PATH/"
    echo "Results archived to $ARCHIVE_PATH"
else
    echo "Skipping archiving (exit code non-zero or archive directory not set/found)."
fi

echo "Job finished."
exit $EXIT_CODE 