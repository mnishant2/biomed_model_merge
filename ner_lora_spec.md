# 🛠️  CODE‑GEN REQUEST – GENERATIVE SPAN‑STRING NER PIPELINE

### High‑level goal

Implement LoRA‑fine‑tuning of an 8 B Llama‑3 OpenBioLLM backbone for generative NER.

The model must ingest a sentence and emit a span string `"<TAG> 0‑7 ; 19‑32"` where each pair is character offsets in the input chunk (no entity type).

Keep one optional token‑classification head (B/I/O) for fast dev metrics; the main loss is still causal‑LM.

Use Unsloth for memory‑ and speed‑optimised LoRA, bf16 weights.

Provide: datamodule, model class, Trainer wrapper, Hydra configs, W&B logging, SLURM sbatch for (a) single run and (b) wandb sweep, plus a stand‑alone vLLM evaluation script.

---

## OVERVIEW

Fine‑tune an **8 B “aaditya/Llama3‑OpenBioLLM‑8B”** backbone with **LoRA (rank 8, bf16)** for *generative* biomedical NER.\
Targets are compact **character‑offset spans** (no types).\
No 4‑bit quantisation; we train in bf16 on 48 GB A100 or 94 GB H100.

---

## DELIVERABLE TREE

```
biomerge/
├── requirements.txt
├── conf/
│   ├── model/unsloth_lora_bf16.yaml
│   ├── task/ner_span.yaml
│   ├── runtime/a100.yaml
│   ├── runtime/h100.yaml
│   ├── sweep/lora_ner_span.yaml
│   └── deepspeed/zero2_offload.json
├── src/
│   ├── datamodules/ner_span.py
│   ├── models/unsloth_lora_ner.py
│   ├── callbacks/ner_metrics.py
│   ├── scripts/eval_vllm.py
│   ├── tests/
│   └── main.py
├── scripts/
│   ├── job_single_ner.sbatch
│   ├── job_single_ner_h100.sbatch
│   ├── job_sweep_ner.sbatch
│   └── run_local_debug.sh
└── README.md
```

---

## DATA ASSUMPTION

- Converted JSONL lives under `data/union_span/ner/{train,dev,test}.jsonl` with:
  ```json
  { "input": "<NER> …", "target": "<TAG> 0‑7 ; 19‑32", … }
  ```

---

## TOKENIZER

- Add tokens once:  `<NER>  <TAG>  <O>  <B>  <I>`
- Save to `tokenizers/llama3_biomerge_tok/`
- Pad token = eos.

Log every metric (loss, span\_F1, optional token\_head\_F1) to W&B.

---

## UNSLOTH MODEL

```python
self.backbone, self.tok = FriendlyModel(
    base_model,
    dtype="bf16",
    quantization=None,
    autotune=True
).setup_for_training(r=8, lora_alpha=16, lora_dropout=0.05,
                     target_modules="all")
```

No 4‑bit. Gradient‑checkpointing ON (`self.backbone.enable_gradient_checkpointing()`).

---

## Data assumptions

```json
{
  "task": "NER",
  "input": "<NER> Aspirin can trigger asthma attacks in sensitive patients .",
  "target": "<TAG> 0-7 ; 19-32",
  "doc_id": "bc5cdr_014",
  "dataset": "bc5cdr"
}
```

Files live in `data/union_span/ner/{train,dev,test}.jsonl`.

Max input length 256 tokens; span string rarely exceeds 64 tokens. → set `max_new_tokens = 64` during generation.

### Tokenizer prep (one‑time)

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("aaditya/Llama3-OpenBioLLM-8B", use_fast=True)
special_tokens = ["<NER>", "<TAG>", "<O>", "<B>", "<I>"]
tok.add_tokens([t for t in special_tokens if t not in tok.vocab])
tok.pad_token = tok.eos_token  # ensure padding exists
tok.save_pretrained("tokenizers/llama3_biomerge_tok/")
```

---

## OPTIONAL TOKEN HEAD (fast dev metrics)

```python
self.token_head = nn.Linear(H, 3)
loss = CE(ignore_index=-100)
```

Only built if `cfg.model.fast_metrics = true`.

---

## Datamodule specification (`src/datamodules/ner_span.py`)

`__getitem__` returns:

```python
{
  "input_ids": …,              # tokenised sentence (<NER> included)
  "attention_mask": …,
  "labels": …,                 # tokenised "<TAG> 0‑7 ; …" padded to max_len
  "token_labels": … or None,   # optional BIO ints for TC head
}
```

- Use `datasets.load_dataset("json", …)` so splits can stream.
- Allow `train_fraction` (e.g. 0.1) for sweep speed‑ups.
- If fast\_metrics=True, build token\_labels on‑the‑fly:
  - convert span pairs → per‑token BIO via word\_ids().



---

## Model class (`src/models/unsloth_lora_ner.py`)

```python
from unsloth import FriendlyModel
import torch.nn as nn, torch

class UnslothLoRANER(nn.Module):
    def __init__(self, base="aaditya/Llama3-OpenBioLLM-8B", r=8, alpha=16,
                 dropout=0.05, add_token_head=True, dtype="bf16"):
        super().__init__()
        self.backbone, self.tokenizer = FriendlyModel(
            base, dtype=dtype, quantization="q4", autotune=True
        ).setup_for_training(r=r, lora_alpha=alpha,
                             lora_dropout=dropout, target_modules="all")
        if add_token_head:
            H = self.backbone.config.hidden_size
            self.token_head = nn.Linear(H, 3)               # O/B/I
            self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.token_head, self.ce = None, None

    def forward(self, input_ids, attention_mask=None,
                labels=None, token_labels=None):
        out = self.backbone(input_ids, attention_mask, labels=labels,
                            output_hidden_states=True)
        loss = out.loss
        if self.token_head is not None and token_labels is not None:
            hidden = out.hidden_states[-1]
            tok_logits = self.token_head(hidden)
            loss_tok = self.ce(tok_logits.view(-1,3), token_labels.view(-1))
            loss = loss + loss_tok
        return {"loss": loss, "logits": out.logits}
```

LoRA params: rank 8, α 16, dropout 0.05.

Generation config defaults: `max_new_tokens=64`, greedy.

---

## Training script (`src/main.py`)

Accept Hydra groups: `task=ner_span model=lora_ner_unsloth runtime=a100`.
Use `torchrun` (`accelerate` is optional) for multi‑GPU.

### TRAINER ARGUMENTS (`conf/runtime/a100.yaml`)

```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
num_train_epochs: 3
learning_rate: 2e-4
load_best_model_at_end: true
metric_for_best_model: eval_ner_f1        # ← computed every eval_steps
greater_is_better: true
lr_scheduler_type: linear
warmup_ratio: 0.05
fp16: false
bf16: true
logging_steps: 50
predict_with_generate: false          # fast path during training
evaluation_strategy: steps
eval_steps: 500
save_steps: 500
save_total_limit: 2
report_to: ["wandb"]
ddp_find_unused_parameters: false
gradient_checkpointing: true
```

CLI flag `--subsample 0.1` passes to datamodule.

---

## Max‑length recommendations

| Stage                | Value       |
| -------------------- | ----------- |
| input\_ids           | 256         |
| labels (span string) | ≤ 64 tokens |
| `max_new_tokens`     | 64          |

---

## Evaluation inside Trainer

- If `fast_metrics=True` **and** token head exists → compute seqeval from token logits.
- Else → greedy `generate()` on dev with `max_new_tokens=64`; parse span strings and compute char‑offset F1 wrt gold span.
- Use the fast, teacher‑forced metric as the “watch metric” while training, at steps 500,1000,1500, with predict_with_generate: False
- Run a greedy‑decode dev check at the end of every epoch and after training finishes
    ```python
    # turn on generation just for this pass
        trainer.args.predict_with_generate = True
        trainer.args.generation_max_new_tokens = 64
        dev_metrics = trainer.evaluate(datamodule.val_dataset,
                                    metric_key_prefix="greedy_dev")
        print(dev_metrics["greedy_dev_ner_f1"])
    ```
If greedy F1 differs from fast F1 by more than ~0.01 you probably have a formatting bug in the span string.
- Final test‑set evaluation & save
    Add to src/main.py

```python
        if cfg.run_test_eval:
            test_metrics = trainer.predict(datamodule.test_dataset,
                                        metric_key_prefix="test")
            trainer.log_metrics("test", test_metrics.metrics)
            trainer.save_metrics("test", test_metrics.metrics)
```
Invoke via:
```bash
        python -m src.main ... +run_test_eval=true
```
This runs greedy decoding (because predict_with_generate is still True) on the test split using the best checkpoint auto‑loaded by Trainer.



### DATALOADER / WORKERS

Hydra flags in task/ner_span.yaml:
```yaml
dataloader:

    num_workers: 8            # overridable
    pin_memory: true
```


---

## DEEPSPEED (optional)

Provide `conf/deepspeed/zero3_offload.json`, but disabled by default. Hydra override: +deepspeed=conf/deepspeed/zero3\_offload.json.

---

## W&B SWEEP (`conf/sweep/lora_ner_span.yaml`)

```yaml
method: grid
metric:
  name: eval_ner_f1
  goal: maximize
parameters:
  learning_rate: {values: [1e-4, 2e-4, 3e-4]}
  lora_rank: {values: [4, 8, 16]}
  subsample: {values: [0.2]}
  Dropout: {values:[0.05, 0.1]}
```

Command array launches `src.main` with overrides.\
```yaml
command:

 \- ${env}

- python

- -m

- src.main

- mode=train

- task=ner_span

- model=lora_ner_unsloth

- runtime=a100

- datamodule.params.root=$TMPDIR/data/union/NER_span

- training.output_dir=$TMPDIR/outputs

- subsample=${subsample}

- learning_rate=${lr}

- lora_ner_unsloth.params.r=${r}

- ${args_no_hyphens}

```
W&B automatic checkpoint comparison in sweeps

Sweep YAML already sets metric.name: eval_ner_f1, goal: maximize.

When each agent finishes it logs the best F1 and the config; W&B sweep dashboard ranks runs, you pick the run with the best dev F1.

Download its checkpoint artifact or locate it in outputs/run‑42‑ner_span/checkpoint‑2000
---
## Early‑stop to save GPU time

```python
from transformers import EarlyStoppingCallback
early_stop = EarlyStoppingCallback(
    early_stopping_patience = 3,      # 3 evals without improvement
    early_stopping_threshold = 0.001  # min delta
)
trainer = Trainer(..., callbacks=[ner_metrics_cb, early_stop])
```
With eval_steps=500, patience 3 ≈ ~½ epoch on your 32 k‑sample dataset.

---
## SLURM JOB – SINGLE TRAIN (`scripts/job_single_ner.sbatch`)

```bash
#!/bin/bash
#SBATCH --job-name=ner_span
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
module purge; module load 2024 cuDNN/9.5.0.50-CUDA-12.6.0 NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
source $HOME/envs/taskmerge/bin/activate
WORKROOT="$HOME/nmishra/biomerge"
PERSIST_DATA="$WORKROOT/data/union"
TMP_DATA="$TMPDIR/data"; TMP_OUT="$TMPDIR/outputs"
mkdir -p "$TMP_DATA/union" "$TMP_OUT"
rsync -a "$PERSIST_DATA/" "$TMP_DATA/"
export WANDB_PROJECT=TaskMerge
srun torchrun --nproc_per_node 1 -m src.main \
     mode=train task=ner_span model=lora_ner_unsloth runtime=a100 \
     datamodule.params.root=$TMPDIR/data/union_span/ner \
     training.output_dir=$TMPDIR/outputs
```

After training: `rsync -a $TMPDIR/outputs/ $ARCHIVE/ner_span_$SLURM_JOB_ID`.

Option --run_test_eval triggers final test‑set evaluation.

---

## LOCAL DEBUG (`scripts/run_local_debug.sh`)

Runs with `--subsample 0.01 --num_train_epochs 0.1`.

---

## TEST SCRIPTS

1. `test_tokenise.py` – ensure datamodule tensors align.
2. `test_forward.py` – forward pass loss finite and ensures loss decreases after one optimisation step
3. `test_generate.py` – `.generate()` returns proper tag string.
4. `test_save_best.py` – mocks three eval steps with ascending F1 and asserts only the highest checkpoint remains when load_best_model_at_end=true
run 
```bash
pytest -q src/tests
```
---

## EVALUATION SCRIPT via vLLM

### Launch server (same GPU as checkpoint)

```bash
python -m vllm.entrypoints.openai.api_server \
       --model outputs/ner_span_bf16/checkpoint-500 \
       --dtype bfloat16 --port 8001 &
```

### `src/scripts/eval_vllm.py`
just for reference
```python
    """Offline span‑F1 evaluation using a running vLLM OpenAI server."""
    import argparse, requests, datasets, tqdm
    from transformers import AutoTokenizer

    def parse_span(txt: str):
        """Return a set of (start,end) ints from a "<TAG> 0-7 ; 19-32" string."""
        txt = txt.replace("<TAG>", "").strip()
        spans = set()
        for part in txt.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                s, e = map(int, part.split("-"))
                spans.add((s, e))
            except ValueError:
                pass
        return spans

    def main(split="dev", port=8001, data_dir="data/union_span/ner"):
        tok = AutoTokenizer.from_pretrained("tokenizers/llama3_biomerge_tok")
        ds = datasets.load_dataset("json", data_files={"data": f"{data_dir}/{split}.jsonl"})["data"]
        TP = FP = FN = 0
        for ex in tqdm.tqdm(ds, desc=f"{split} eval"):
            prompt = ex["input"] + " <TAG>"
            resp = requests.post(
                f"http://localhost:{port}/v1/completions",
                json={"prompt": prompt, "max_tokens": 64, "temperature": 0}, timeout=120,
            )
            pred = parse_span(resp.json()["choices"][0]["text"])
            gold = parse_span(ex["target"])
            TP += len(pred & gold)
            FP += len(pred - gold)
            FN += len(gold - pred)
        f1 = 2 * TP / (2 * TP + FP + FN) if TP else 0.0
        print(f"Span F1 = {f1:.4f}")

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--split", default="dev")
        parser.add_argument("--port", type=int, default=8001)
        parser.add_argument("--data_dir", default="data/union_span/ner")
        main(**vars(parser.parse_args()))
```

Run:

```bash
python src/scripts/eval_vllm.py --split dev --port 8001
```

Evaluates **≈ 800 sentences in < 60 s on an A100**.\
Expect span‑F1 to match training metrics (≥ 0.75 at convergence).

---

Produce:

- src/datamodules/ner\_span.py

- src/models/unsloth\_lora\_ner.py

- src/callbacks/ner\_metrics.py (updated)

- conf/model/lora\_ner\_unsloth.yaml, conf/task/ner\_span.yaml, conf/runtime/a100.yaml

- SLURM scripts job\_single\_ner.sbatch, job\_sweep\_ner.sbatch

- scripts/eval\_vllm.py

- README.md snippet with training + evaluation commands.

And all other files in the structure described above

## Acceptance criteria

- `pytest -q tests` passes.
- `run_local_debug.sh` finishes in < 3 min on CPU.
- A100 training job reaches dev F1 > 0.75 within 2 epochs.
- **`eval_vllm.py`**\*\* prints span‑F1 within ±0.02 of Trainer metric.\*\*

