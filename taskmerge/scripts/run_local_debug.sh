#!/bin/bash

# Script to run a quick local debug training run

echo "Starting local debug run..."

# Activate your python environment if necessary
# source /path/to/your/venv/bin/activate

# Ensure the prepared tokenizer exists
TOKENIZER_PATH="taskmerge/tokenizers/llama3_biomerge_tok"
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "Tokenizer not found at $TOKENIZER_PATH. Running prepare_tokenizer.py..."
    python taskmerge/scripts/prepare_tokenizer.py || exit 1
fi

# Ensure dummy data exists for local testing (using the main script's logic)
# The main script's __main__ block for the datamodule creates dummy data if needed,
# but it's safer to ensure it here or point to real small data.
# For simplicity, we rely on the datamodule test logic or assume small data exists.
DATA_PATH="taskmerge/data/union_span/ner"
echo "Assuming data exists at $DATA_PATH (or will be created by datamodule test logic if run directly)"

# Run the main training script with overrides for a short debug run
# Use python directly instead of torchrun for single-process debug
python taskmerge/src/main.py \
    task=ner_span \
    model=unsloth_lora_bf16 \
    runtime=a100 `# Use A100 config as base, overrides below matter more` \
    ++runtime.training.per_device_train_batch_size=2 `# Small batch` \
    ++runtime.training.gradient_accumulation_steps=1 \
    ++runtime.training.num_train_epochs=0.1 `# Train for a fraction of an epoch` \
    ++runtime.training.eval_steps=10 `# Evaluate frequently` \
    ++runtime.training.logging_steps=5 \
    ++runtime.training.save_steps=20 `# Save less often` \
    ++runtime.training.save_total_limit=1 \
    ++runtime.training.report_to=[] `# Disable W&B for local debug` \
    ++runtime.training.load_best_model_at_end=false `# Disable loading best model` \
    ++task.datamodule.train_fraction=0.01 `# Use tiny fraction of data` \
    ++task.datamodule.num_workers=0 `# Easier debugging` \
    ++task.evaluation.run_test_eval=false `# Don't run test eval` \
    # Add other overrides as needed, e.g.:
    # model.add_token_head=false # Disable token head if testing base LM

echo "Local debug run finished." 