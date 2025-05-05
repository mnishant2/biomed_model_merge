#!/bin/bash
#SBATCH --job-name=ner_span_a100 # Changed name slightly for clarity
#SBATCH --partition=gpu           # Adjust partition name if needed
#SBATCH --gres=gpu:a100:1       # Request 1 A100 GPU
#SBATCH --time=04:00:00         # Max runtime
#SBATCH --cpus-per-task=16      # Request CPUs
#SBATCH --mem=96G             # Request Memory
#SBATCH --output=logs/ner_span_a100-%j.out # Log file location
#SBATCH --error=logs/ner_span_a100-%j.err  # Error file location

# --- Environment Setup ---
echo "Loading modules..."
module purge
# Adjust module load command based on your HPC environment
# Example provided in spec:
module load 2024 cuDNN/9.5.0.50-CUDA-12.6.0 NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
echo "Modules loaded."

# Activate Python environment
# Adjust the path to your virtual environment
ENV_PATH="$HOME/envs/taskmerge/bin/activate" # Example from spec
if [ -f "$ENV_PATH" ]; then
    echo "Activating Python environment: $ENV_PATH"
    source "$ENV_PATH"
else
    echo "Python environment not found at $ENV_PATH. Using default python."
fi

# --- Variables and Paths ---
# Adjust WORKROOT if your project is located elsewhere relative to $HOME
WORKROOT="$HOME/data/storage_hpc_nishant/taskmerge" # Corrected base path
PERSIST_DATA_ROOT="$WORKROOT/data/union_span/ner" # Path to persistent data
PERSIST_TOKENIZER_ROOT="$WORKROOT/tokenizers/llama3_biomerge_tok" # Path to persistent tokenizer
# Use TMPDIR for fast node-local storage if available, otherwise use WORKROOT
SCRATCH_DIR=${TMPDIR:-$WORKROOT/scratch/$SLURM_JOB_ID} # Use TMPDIR or create a job-specific scratch
TMP_DATA="$SCRATCH_DIR/data/union_span/ner"
TMP_TOKENIZER="$SCRATCH_DIR/tokenizers/llama3_biomerge_tok"
TMP_OUT="$SCRATCH_DIR/outputs"
ARCHIVE_DIR="$WORKROOT/archive" # Optional: Define archive location

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
export WANDB_PROJECT=${WANDB_PROJECT:-TaskMerge} # Use environment variable or default
export WANDB_DIR=$TMP_OUT # Store wandb logs in scratch output
echo "WANDB_PROJECT set to: $WANDB_PROJECT"

# --- Run Training ---
echo "Starting training run..."
cd "$WORKROOT" # Ensure we are in the project root directory

# Use torchrun for potential multi-GPU scaling (even if starting with 1)
# Adjust nproc_per_node if using more GPUs
SRUN_ARGS=(
    "srun"
    "--cpu-bind=none"
    # Add other srun options if necessary
)
TORCHRUN_ARGS=(
    "torchrun"
    "--nproc_per_node=1" # Match --gres gpu count
    "--nnodes=1"
    "-m"
    "taskmerge.src.main" # Run main module
)
HYDRA_ARGS=(
    "task=ner_span"
    "model=unsloth_lora_bf16"
    "runtime=a100"
    "task.datamodule.data_dir=$TMP_DATA"             # Point datamodule to scratch data
    "model.tokenizer_path=$TMP_TOKENIZER"         # Point model to scratch tokenizer
    "runtime.training.output_dir=$TMP_OUT"        # Point trainer output to scratch outputs
    # Add +task.evaluation.run_test_eval=true here if you want test set eval by default
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
    ARCHIVE_PATH="$ARCHIVE_DIR/ner_span_a100_$SLURM_JOB_ID"
    mkdir -p "$ARCHIVE_PATH"
    rsync -a --info=progress2 "$TMP_OUT/" "$ARCHIVE_PATH/"
    echo "Results archived to $ARCHIVE_PATH"
else
    echo "Skipping archiving (exit code non-zero or archive directory not set/found)."
fi

# Clean up scratch space if needed (be careful with this on shared systems)
# echo "Cleaning up scratch space: $SCRATCH_DIR"
# rm -rf "$SCRATCH_DIR"

echo "Job finished."
exit $EXIT_CODE 