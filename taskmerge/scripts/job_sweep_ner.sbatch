#!/bin/bash
#SBATCH --job-name=ner_sweep_agent
#SBATCH --partition=gpu           # Adjust partition name
#SBATCH --gres=gpu:a100:1       # GPU type/count per agent
#SBATCH --time=04:00:00         # Max time per agent run
#SBATCH --cpus-per-task=16      # CPUs per agent
#SBATCH --mem=96G             # Memory per agent
#SBATCH --output=logs/ner_sweep_agent-%j.out
#SBATCH --error=logs/ner_sweep_agent-%j.err

# --- Environment Setup ---
echo "Loading modules..."
module purge
module load 2024 cuDNN/9.5.0.50-CUDA-12.6.0 NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
echo "Modules loaded."

# Activate Python environment
ENV_PATH="$HOME/envs/taskmerge/bin/activate"
if [ -f "$ENV_PATH" ]; then
    echo "Activating Python environment: $ENV_PATH"
    source "$ENV_PATH"
else
    echo "Python environment not found at $ENV_PATH. Using default python."
fi

# --- Variables and Paths ---
WORKROOT="$HOME/data/storage_hpc_nishant/taskmerge"
# Data/Tokenizer should be accessible from where agent runs (e.g., shared filesystem)
# No need to copy to TMPDIR here as the sweep config points to persistent paths

# --- W&B Agent Setup ---
# The SWEEP_ID needs to be obtained after creating the sweep via:
# wandb sweep conf/sweep/lora_ner_span.yaml
# The command output will contain the sweep ID like: wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
# Replace YOUR_SWEEP_ID below or pass it as an argument/environment variable
SWEEP_ID="${WANDB_SWEEP_ID:-YOUR_SWEEP_ID}"
WANDB_PROJECT=${WANDB_PROJECT:-TaskMerge}
WANDB_ENTITY=${WANDB_ENTITY:-YOUR_ENTITY} # Replace with your W&B entity/username

if [ "$SWEEP_ID" == "YOUR_SWEEP_ID" ]; then
    echo "Error: SWEEP_ID is not set. Please replace YOUR_SWEEP_ID or set WANDB_SWEEP_ID environment variable."
    exit 1
fi
if [ "$WANDB_ENTITY" == "YOUR_ENTITY" ]; then
    echo "Error: WANDB_ENTITY is not set. Please replace YOUR_ENTITY or set WANDB_ENTITY environment variable."
    exit 1
fi

AGENT_CMD="wandb agent ${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"

# --- Run W&B Agent ---
echo "Starting W&B agent for sweep: $SWEEP_ID"
echo "Agent command: $AGENT_CMD"
cd "$WORKROOT" # Run agent from the project root

# Run the agent (it will pick up jobs from the sweep queue)
# srun might not be necessary if wandb agent doesn't need specific binding
$AGENT_CMD
EXIT_CODE=$?

echo "W&B agent finished with exit code: $EXIT_CODE"
exit $EXIT_CODE 