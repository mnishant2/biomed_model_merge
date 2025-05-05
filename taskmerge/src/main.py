# Placeholder for main training script (Hydra + Trainer) 

import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback
)
import datasets
import wandb

# Assuming your modules are structured like this
from taskmerge.src.datamodules.ner_span import NERSpanDataModule
from taskmerge.src.models.unsloth_lora_ner import UnslothLoRANER
from taskmerge.src.callbacks.ner_metrics import NerMetricsCallback, parse_span_from_string

# Add src directory to path if needed for Hydra instantiation
# (Might not be necessary depending on execution context)
# script_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.dirname(script_dir)
# sys.path.append(src_dir)

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("get_lora_target_modules", lambda modules_str: modules_str.split(",") if isinstance(modules_str, str) else modules_str)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("---------------- Configuration ---------------")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------------------------")

    # --- Environment Setup ---
    set_seed(cfg.runtime.training.seed if hasattr(cfg.runtime.training, 'seed') else 42)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Hydra output directory: {output_dir}")
    # Update runtime output_dir to actual hydra path for Trainer
    with open_dict(cfg): # Allow modifications
         cfg.runtime.training.output_dir = output_dir

    # --- W&B Setup ---
    if "wandb" in cfg.runtime.training.report_to:
        try:
            import wandb
            wandb_config = cfg.runtime.wandb
            wandb.init(
                project=wandb_config.project,
                name=hydra.core.hydra_config.HydraConfig.get().job.name,
                entity=wandb_config.get("entity", None),
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                dir=output_dir, # Log wandb files within hydra output dir
                resume="allow", # Allow resuming runs
                # id=wandb_run_id # Optionally set for resuming
            )
            logger.info("WandB initialized.")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging.")
            cfg.runtime.training.report_to = [r for r in cfg.runtime.training.report_to if r != "wandb"]
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            cfg.runtime.training.report_to = [r for r in cfg.runtime.training.report_to if r != "wandb"]

    # --- Load Tokenizer (primarily for collator and callback) ---
    # Model and Datamodule load their own instances
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path, use_fast=True)
        if tokenizer.pad_token is None:
            logger.warning("Setting pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {cfg.model.tokenizer_path}: {e}")
        return # Cannot proceed without tokenizer

    # --- Instantiate Datamodule ---
    logger.info(f"Instantiating datamodule <{cfg.task.datamodule._target_}>")
    # Handle subsample argument if passed via CLI (overrides yaml)
    if 'subsample' in cfg:
         logger.info(f"Overriding train_fraction with CLI --subsample: {cfg.subsample}")
         cfg.task.datamodule.train_fraction = cfg.subsample
    datamodule: NERSpanDataModule = hydra.utils.instantiate(cfg.task.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="fit") # Setup train and validation datasets

    # --- Instantiate Model ---
    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: UnslothLoRANER = hydra.utils.instantiate(cfg.model)

    # Ensure model uses the same tokenizer instance with correct padding
    model.tokenizer = tokenizer
    model.backbone.config.pad_token_id = tokenizer.pad_token_id

    # --- Data Collator --- #
    # Handles padding within the batch for sequence-to-sequence tasks
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model.backbone, # Pass the underlying HF model
        label_pad_token_id=-100, # Pad labels with -100
        pad_to_multiple_of=8     # Optimize padding for tensor cores
    )

    # --- Callbacks ---
    callbacks = []
    # 1. Early Stopping
    if cfg.runtime.training.get("early_stopping_patience"):
        early_stop_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.runtime.training.early_stopping_patience,
            early_stopping_threshold=cfg.runtime.training.get("early_stopping_threshold", 0.001)
        )
        callbacks.append(early_stop_callback)
        logger.info(f"Added EarlyStoppingCallback (patience={cfg.runtime.training.early_stopping_patience})")

    # 2. NER Metrics Callback (Handles both token and generative metrics)
    # We need the raw validation dataset for generative evaluation
    ner_metrics_callback = NerMetricsCallback(
        eval_dataset=datamodule.val_dataset.dataset if isinstance(datamodule.val_dataset, torch.utils.data.Subset) else datamodule.val_dataset,
        tokenizer=tokenizer,
        fast_metrics_enabled=cfg.task.evaluation.use_fast_metrics,
        gen_eval_enabled=True, # Always enable capability, triggering depends on Trainer args
        max_new_tokens=cfg.task.evaluation.max_new_tokens,
        metric_key_prefix="eval", # Prefix for validation metrics
        device=torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    )
    callbacks.append(ner_metrics_callback)
    logger.info("Added NerMetricsCallback.")

    # --- Training Arguments --- #
    logger.info("Instantiating TrainingArguments...")
    training_args = TrainingArguments(
        # Pass configuration directly from OmegaConf dict
        **OmegaConf.to_container(cfg.runtime.training, resolve=True)
    )

    # --- Trainer --- #
    logger.info("Instantiating Trainer...")
    trainer = Trainer(
        model=model.backbone, # Pass the PEFT model (Unsloth handles nn.Module wrapping)
        args=training_args,
        train_dataset=datamodule.train_dataset,
        eval_dataset=datamodule.val_dataset,
        tokenizer=tokenizer, # Pass tokenizer for generation
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ner_metrics_callback.compute_metrics_for_trainer # Use callback method
    )

    # --- Training --- #
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=cfg.runtime.training.get("resume_from_checkpoint", None))
    trainer.save_model()  # Saves the LoRA adapter weights
    logger.info("Training finished.")

    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # --- Final Evaluation (Validation Set) ---
    # Uses the best model checkpoint loaded at end
    # Includes both fast (if enabled) and generative metrics
    logger.info("Performing final evaluation on validation set...")

    # Force generation for final eval pass as per spec
    trainer.args.predict_with_generate = True
    trainer.args.generation_max_new_tokens = cfg.task.evaluation.max_new_tokens

    # Re-run evaluate to get metrics from the best checkpoint with generation
    # Note: NerMetricsCallback's compute_metrics won't easily compute generative F1 here.
    # We rely on the fact that the callback's on_evaluate hook (triggered by evaluate)
    # *does* compute and log the generative metrics (`eval_ner_f1`, etc.) directly.
    final_eval_metrics = trainer.evaluate(eval_dataset=datamodule.val_dataset, metric_key_prefix="eval_final")

    logger.info(f"Final Validation Metrics: {final_eval_metrics}")
    trainer.log_metrics("eval_final", final_eval_metrics)
    trainer.save_metrics("eval_final", final_eval_metrics)

    # Specifically log the generative F1 if available
    gen_f1_key = "eval_final_ner_f1"
    if gen_f1_key in final_eval_metrics:
        logger.info(f"Final Validation Generative Span F1: {final_eval_metrics[gen_f1_key]:.4f}")
        if wandb.run:
            wandb.summary[gen_f1_key] = final_eval_metrics[gen_f1_key]
    else:
        logger.warning(f"Metric '{gen_f1_key}' not found in final eval metrics. Check callback logging.")

    # Compare fast vs generative F1 if both ran
    fast_f1_key = "eval_final_token_f1"
    if gen_f1_key in final_eval_metrics and fast_f1_key in final_eval_metrics:
         diff = abs(final_eval_metrics[gen_f1_key] - final_eval_metrics[fast_f1_key])
         logger.info(f"Token F1 vs Span F1 diff: {diff:.4f}")
         if diff > 0.02: # Check against threshold from spec
             logger.warning("Difference between token F1 and generative span F1 > 0.02! Check implementation.")

    # --- Test Set Evaluation (Optional) --- #
    if cfg.task.evaluation.run_test_eval:
        logger.info("Performing evaluation on test set...")
        datamodule.setup(stage="test") # Setup test dataset
        test_metrics_callback = NerMetricsCallback(
            eval_dataset=datamodule.test_dataset.dataset if isinstance(datamodule.test_dataset, torch.utils.data.Subset) else datamodule.test_dataset,
            tokenizer=tokenizer,
            fast_metrics_enabled=cfg.task.evaluation.use_fast_metrics,
            gen_eval_enabled=True,
            max_new_tokens=cfg.task.evaluation.max_new_tokens,
            metric_key_prefix="test",
            device=torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        )
        # Use predict to get metrics, ensuring predict_with_generate is True
        trainer.args.predict_with_generate = True
        test_results = trainer.predict(datamodule.test_dataset, metric_key_prefix="test")
        test_metrics = test_results.metrics

        # Additionally, compute generative metrics using the callback explicitly
        # as predict doesn't easily give us the raw dataset access needed inside compute_metrics
        logger.info("Computing generative metrics on test set explicitly...")
        test_gen_metrics = test_metrics_callback._compute_generative_metrics(model.backbone, test_metrics_callback.eval_dataset)
        test_metrics.update(test_gen_metrics) # Add generative metrics

        logger.info(f"Test Metrics: {test_metrics}")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        if wandb.run:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            if "test_ner_f1" in test_metrics:
                 wandb.summary["test_ner_f1"] = test_metrics["test_ner_f1"]

    logger.info("Script finished successfully.")
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    # Setup basic logging for potential issues before hydra starts
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    main() 