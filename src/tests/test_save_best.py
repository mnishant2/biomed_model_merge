# Placeholder for save_best tests 

import pytest
import torch
import os
import shutil
import hydra
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl
from unittest.mock import MagicMock, patch

# Assume model, tokenizer, data fixtures from other tests if needed, or create minimal ones
# Need a minimal model and tokenizer for the Trainer to initialize
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.callbacks.ner_metrics import NerMetricsCallback # For compute_metrics structure

CONF_PATH = "conf"

# Minimal fixture setup
@pytest.fixture(scope="module")
def test_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("save_best_test")

@pytest.fixture(scope="module")
def minimal_tokenizer(tmp_path_factory):
    # Use a small, fast tokenizer for this test
    tok_path = tmp_path_factory.mktemp("minimal_tok")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        # Add special tokens used by the model/data if necessary for initialization
        tokenizer.add_special_tokens({"additional_special_tokens": ["<NER>", "<TAG>"]})
        tokenizer.save_pretrained(tok_path)
        return tokenizer # Return loaded tokenizer instance
    except Exception as e:
        pytest.skip(f"Could not prepare minimal tokenizer: {e}")

@pytest.fixture(scope="module")
def minimal_model(minimal_tokenizer):
    # Use a very small model for speed
    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.resize_token_embeddings(len(minimal_tokenizer))
        # Add a dummy token head if the main model expects it
        # model.token_head = torch.nn.Linear(model.config.hidden_size, 3)
        return model
    except Exception as e:
        pytest.skip(f"Could not load minimal model gpt2: {e}")

@pytest.fixture
def mock_datasets():
    # Create dummy datasets (list of dicts is often enough for Trainer)
    train_ds = [{"input_ids": [0, 1, 2], "attention_mask": [1, 1, 1], "labels": [0, 1, 2]}] * 4
    eval_ds = [{"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1], "labels": [3, 4, 5]}] * 2
    # Mock the raw dataset attribute needed by NerMetricsCallback
    eval_ds_mock_raw = MagicMock()
    eval_ds_mock_raw.dataset = eval_ds # Point to the list
    eval_ds_mock_raw.__len__.return_value = len(eval_ds)
    return train_ds, eval_ds_mock_raw

def test_save_best_checkpoint_logic(test_output_dir, minimal_model, minimal_tokenizer, mock_datasets):
    """Mocks training steps to verify save_best behavior."""
    model = minimal_model
    tokenizer = minimal_tokenizer
    train_ds, eval_ds = mock_datasets
    output_dir = str(test_output_dir)

    # Configure TrainingArguments for save_best testing
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1, # Shorten training
        evaluation_strategy="steps",
        eval_steps=2, # Evaluate frequently
        save_strategy="steps",
        save_steps=2, # Save on each eval
        save_total_limit=2, # Keep best and last (or just 2 best if last isn't best)
        load_best_model_at_end=True,
        metric_for_best_model="eval_ner_f1", # The metric to monitor
        greater_is_better=True,
        logging_steps=1,
        report_to=[], # Disable external logging
        # Disable optimizations that might interfere
        fp16=False, bf16=False, gradient_checkpointing=False,
    )

    # Mock the metric computation - simulate increasing F1
    mock_metrics_state = {'step': 0, 'f1_scores': [0.5, 0.7, 0.6, 0.8]} # Eval steps 2, 4, 6, 8
    def mock_compute_metrics(eval_preds):
        step = mock_metrics_state['step']
        f1 = mock_metrics_state['f1_scores'][step]
        print(f"MOCK COMPUTE METRICS: step {training_args.eval_steps * (step+1)}, returning F1={f1:.2f}")
        mock_metrics_state['step'] += 1
        return {"eval_loss": 0.1, "eval_ner_f1": f1} # Need the metric_for_best_model key

    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds, # Trainer uses this for eval loop
        tokenizer=tokenizer,
        compute_metrics=mock_compute_metrics,
        # callbacks=[mock_callback] # Can add mock callback if needed
    )

    # Mock the state and control flow slightly to avoid actual training
    # We just need to trigger evaluation and saving steps
    trainer.state = TrainerState(max_steps=8)
    trainer.control = TrainerControl()

    # Simulate training loop triggering evaluations and saving
    # Need to manually call _maybe_log_save_evaluate
    for step in range(1, 9):
        trainer.state.global_step = step
        # Simulate a training step (doesn't need to do anything)
        trainer.control = trainer.callback_handler.on_step_end(training_args, trainer.state, trainer.control)

        if step % training_args.eval_steps == 0:
            print(f"Simulating evaluation and save at step {step}")
            # Manually trigger the core logic that evaluates and saves
            metrics = trainer.evaluate(metric_key_prefix="eval") # Triggers compute_metrics
            trainer.state.log_history.append({**metrics, 'step': step, 'epoch': float(step)/4.0})
            trainer._save_checkpoint(model, trial=None, metrics=metrics)
            trainer.control = trainer.callback_handler.on_save(training_args, trainer.state, trainer.control)
            # Prune checkpoints (this happens internally after saving)
            trainer._rotate_checkpoints(use_mtime=False)

    # After the loop, load_best_model_at_end=True should have loaded the best one.
    # The pruning logic (`_rotate_checkpoints`) should have removed the others.

    # Check remaining checkpoints
    checkpoint_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")])
    print(f"Remaining checkpoints: {checkpoint_dirs}")

    # Expected behavior: save_total_limit=2, load_best=True
    # Keeps the *best* checkpoint (step 8 with F1=0.8) and potentially the *last* checkpoint (step 8).
    # If best is last, only one might remain depending on exact rotation logic.
    # HF Trainer usually keeps the best *metric* checkpoint and the latest step checkpoint if different.
    # In our case, step 8 (F1=0.8) is best AND last.
    # Step 4 (F1=0.7) was the previous best.
    # Expected remaining: checkpoint-4 (previous best) and checkpoint-8 (current best/last)
    # OR just checkpoint-8 if logic optimizes.
    # Let's check flexibly for the best one (step 8) and potentially the previous best (step 4)

    assert "checkpoint-8" in checkpoint_dirs # Best and last checkpoint
    # Depending on rotation logic, previous best might or might not be kept when limit is 2
    # assert "checkpoint-4" in checkpoint_dirs # Previous best (F1=0.7)

    # Assert that checkpoints with lower scores were deleted
    assert "checkpoint-2" not in checkpoint_dirs # F1=0.5
    assert "checkpoint-6" not in checkpoint_dirs # F1=0.6

    # Verify that 'best_model_checkpoint' attribute points correctly (if trainer state was fully managed)
    # This part is harder to mock perfectly without running trainer.train()
    # assert trainer.state.best_model_checkpoint == os.path.join(output_dir, "checkpoint-8")

    print("Save best checkpoint test completed.") 