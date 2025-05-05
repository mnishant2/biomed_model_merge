# Generative Span NER with Llama3-OpenBioLLM and LoRA

This project implements LoRA fine-tuning of `aaditya/Llama3-OpenBioLLM-8B` for generative Named Entity Recognition (NER). The model learns to predict character offset spans for entities in a given input sentence, formatted as `<TAG> start1-end1 ; start2-end2 ; ...`.

It uses Unsloth for optimized training, PyTorch Lightning via the Transformers `Trainer`, Hydra for configuration management, and Weights & Biases for experiment tracking.

## Project Structure

```
taskmerge/ # Renamed from biomerge as per user request
├── requirements.txt
├── conf/                  # Hydra configuration files
│   ├── config.yaml        # Main config entry point
│   ├── model/unsloth_lora_bf16.yaml
│   ├── task/ner_span.yaml
│   ├── runtime/a100.yaml
│   ├── runtime/h100.yaml
│   ├── sweep/lora_ner_span.yaml
│   └── deepspeed/zero2_offload.json (Optional)
├── src/
│   ├── datamodules/ner_span.py       # PyTorch Lightning DataModule
│   ├── models/unsloth_lora_ner.py    # Model definition (Unsloth + LoRA)
│   ├── callbacks/ner_metrics.py      # Callback for computing metrics
│   ├── scripts/eval_vllm.py          # Standalone evaluation script (vLLM)
│   ├── tests/                        # Pytest tests
│   │   ├── test_tokenise.py
│   │   ├── test_forward.py
│   │   ├── test_generate.py
│   │   └── test_save_best.py
│   └── main.py                       # Main training script (Hydra + Trainer)
├── scripts/                 # Helper & Job scripts
│   ├── prepare_tokenizer.py  # One-time tokenizer setup
│   ├── job_single_ner.sbatch # SLURM script for A100
│   ├── job_single_ner_h100.sbatch # SLURM script for H100
│   ├── job_sweep_ner.sbatch    # SLURM script for W&B sweep agent
│   └── run_local_debug.sh    # Local debug script
├── tokenizers/              # Saved tokenizer files
│   └── llama3_biomerge_tok/
├── data/                    # Data directory (assumed structure)
│   └── union_span/ner/
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── test.jsonl
├── logs/                    # SLURM output logs
├── outputs/                 # Default output dir for local runs/Trainer artifacts
├── archive/                 # Optional persistent storage for SLURM job outputs
└── README.md
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd taskmerge
    ```

2.  **Create a Python environment:** (Recommended)
    ```bash
    python -m venv env
    source env/bin/activate
    # Or using Conda:
    # conda create -n taskmerge python=3.10
    # conda activate taskmerge
    ```

3.  **Install dependencies:**
    *Ensure you have CUDA toolkit compatible with PyTorch/Unsloth installed.*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare the Tokenizer:**
    This script downloads the base tokenizer, adds special tokens (`<NER>`, `<TAG>`, `<O>`, `<B>`, `<I>`), sets the padding token, and saves it locally.
    ```bash
    python taskmerge/scripts/prepare_tokenizer.py
    ```

5.  **Prepare the Data:**
    Place your training, validation, and test data in `taskmerge/data/union_span/ner/` with the following format (one JSON object per line):
    ```json
    {"task": "NER", "input": "<NER> Aspirin can trigger asthma...", "target": "<TAG> 0-7 ; 19-32", ...}
    ```
    Files must be named `train.jsonl`, `dev.jsonl`, and `test.jsonl`.

6.  **Weights & Biases Login (Optional):**
    If using W&B logging (`report_to: ["wandb"]` in runtime configs), log in:
    ```bash
    wandb login
    ```

## Running Tests

Run all tests from the project root directory (`taskmerge/`):
```bash
pytest -q src/tests
```
*Note: Some tests require CUDA and a prepared tokenizer.*

## Local Debugging

Run a short training loop locally on a small fraction of the data without W&B logging:
```bash
bash scripts/run_local_debug.sh
```
This is useful for checking if the pipeline runs without errors before submitting to SLURM.

## Training

### Single Run (SLURM)

Submit a training job to SLURM using the provided scripts. These scripts handle copying data to node-local storage (`$TMPDIR`) if available.

*   **A100:**
    ```bash
    sbatch scripts/job_single_ner.sbatch
    ```
*   **H100:**
    ```bash
    sbatch scripts/job_single_ner_h100.sbatch
    ```

These scripts use `torchrun` and the configurations defined in `conf/`. Checkpoints and logs will be saved to the specified temporary directory (`$TMPDIR/outputs` or `scratch/.../outputs`) and optionally archived to `taskmerge/archive/` on successful completion.

To enable final evaluation on the test set after training, add `+task.evaluation.run_test_eval=true` to the `HYDRA_ARGS` array within the `.sbatch` script.

### W&B Hyperparameter Sweep (SLURM)

1.  **Create the sweep on W&B:**
    Run this command locally or on a login node. It will print a SWEEP_ID.
    ```bash
    wandb sweep taskmerge/conf/sweep/lora_ner_span.yaml
    ```

2.  **Update the Agent Script:**
    Edit `taskmerge/scripts/job_sweep_ner.sbatch` and replace `YOUR_SWEEP_ID` and `YOUR_ENTITY` with the actual sweep ID and your W&B entity (username or team name).

3.  **Launch Agent(s):**
    Submit one or more agent jobs to SLURM. Each job will run one trial of the sweep.
    ```bash
    # Submit one agent
    sbatch scripts/job_sweep_ner.sbatch

    # Submit multiple agents (e.g., 4) using a job array
    # sbatch --array=1-4 scripts/job_sweep_ner.sbatch
    ```
    Agents will execute the command defined in `conf/sweep/lora_ner_span.yaml`, running `src/main.py` with different hyperparameter combinations.

## Evaluation (Standalone with vLLM)

Evaluate a trained checkpoint (LoRA weights merged or adapter loaded) using a dedicated vLLM server for fast inference.

1.  **Launch vLLM Server:**
    Run this on a machine with a GPU where the checkpoint is accessible. Replace `/path/to/checkpoint` with the actual path to your saved checkpoint directory (e.g., `outputs/<hydra_run_dir>/checkpoint-XXXX` or the archived path).
    ```bash
    python -m vllm.entrypoints.openai.api_server \
           --model /path/to/checkpoint \
           --tokenizer taskmerge/tokenizers/llama3_biomerge_tok `# Path to prepared tokenizer` \
           --dtype bfloat16 `# Or float16` \
           --port 8001 `# Port for the server` \
           --max-model-len 2048 `# Adjust based on model/needs` &
    ```
    *Note: Ensure the model path points to a directory containing the merged weights or where the adapter can be loaded.* vLLM might require specific formats or additional arguments depending on how LoRA weights were saved/merged.

2.  **Run Evaluation Script:**
    Once the server is running, execute the evaluation script:
    ```bash
    python taskmerge/src/scripts/eval_vllm.py \
           --split dev `# Or test` \
           --port 8001 `# Match server port` \
           --data_dir taskmerge/data/union_span/ner `# Path to evaluation data`
    ```
    The script will query the vLLM server for each example in the specified split and compute the span-based Precision, Recall, and F1 score.
