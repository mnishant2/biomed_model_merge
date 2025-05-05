# Placeholder for NER Metrics Callback 

import torch
import numpy as np
from transformers import TrainerCallback, TrainerState, TrainingArguments, TrainerControl
from transformers.integrations import WandbCallback
from datasets import load_metric
import logging
from typing import Set, Tuple
import re

logger = logging.getLogger(__name__)

def parse_span_from_string(txt: str) -> Set[Tuple[int, int]]:
    """Parses a generated span string like '<TAG> 0-7 ; 19-32' into a set of (start, end) tuples.
       Handles potential variations and errors gracefully.
    """
    spans = set()
    # Remove potential initial tags and strip whitespace
    txt = re.sub(r"^<TAG>\s*", "", txt).strip()

    # Handle empty string case
    if not txt:
        return spans

    for part in txt.split(";"):
        part = part.strip()
        if not part:
            continue
        # Use regex for more robust parsing of "start-end"
        match = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
        if match:
            try:
                s, e = int(match.group(1)), int(match.group(2))
                # Basic validation: ensure start <= end
                if s <= e:
                    spans.add((s, e))
                else:
                    logger.warning(f"Invalid span format (start > end): '{part}' in '{txt}'")
            except ValueError:
                # Should not happen with regex match, but as safety
                logger.warning(f"Could not parse span part after regex match: '{part}' in '{txt}'")
        else:
             logger.warning(f"Could not parse span part (regex mismatch): '{part}' in '{txt}'")

    return spans

class NerMetricsCallback(TrainerCallback):
    """Callback to compute NER metrics during training and evaluation.

    Handles both:
    1. Fast token-level BIO F1 (if model has token_head and labels are provided).
    2. Generative span-level F1 using greedy decoding.
    """
    def __init__(self, eval_dataset, tokenizer, fast_metrics_enabled=True, gen_eval_enabled=True, max_new_tokens=64, metric_key_prefix="eval", device="cpu"):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.fast_metrics_enabled = fast_metrics_enabled
        self.gen_eval_enabled = gen_eval_enabled
        self.max_new_tokens = max_new_tokens
        self.metric_key_prefix = metric_key_prefix
        self.device = device

        if self.fast_metrics_enabled:
            self.seqeval_metric = load_metric("seqeval")
            self.bio_map_inv = {0: "O", 1: "B-SPAN", 2: "I-SPAN"} # Assuming O=0, B=1, I=2
            logger.info("NerMetricsCallback initialized with fast token metrics (seqeval).")
        if self.gen_eval_enabled:
             logger.info(f"NerMetricsCallback initialized with generative span metrics (max_new_tokens={max_new_tokens}).")

    def _compute_token_metrics(self, predictions, labels):
        # Remove ignored index (-100)
        true_predictions = [
            [self.bio_map_inv[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.bio_map_inv[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        try:
            results = self.seqeval_metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
            return {
                f"{self.metric_key_prefix}_token_precision": results["overall_precision"],
                f"{self.metric_key_prefix}_token_recall": results["overall_recall"],
                f"{self.metric_key_prefix}_token_f1": results["overall_f1"],
                f"{self.metric_key_prefix}_token_accuracy": results["overall_accuracy"],
            }
        except Exception as e:
             logger.error(f"Error computing seqeval metrics: {e}", exc_info=True)
             return {
                 f"{self.metric_key_prefix}_token_precision": 0.0,
                 f"{self.metric_key_prefix}_token_recall": 0.0,
                 f"{self.metric_key_prefix}_token_f1": 0.0,
                 f"{self.metric_key_prefix}_token_accuracy": 0.0,
             }

    def _compute_generative_metrics(self, model, dataset):
        tp, fp, fn = 0, 0, 0
        decode_errors = 0
        total_examples = 0

        # Ensure model is in eval mode
        is_training = model.training
        model.eval()

        logger.info(f"Starting generative evaluation on {len(dataset)} examples...")
        for i, example in enumerate(dataset):
            total_examples += 1
            input_ids = example["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = example["attention_mask"].unsqueeze(0).to(self.device)
            gold_span_str = example["target"] # This should be the raw target string from the dataset

            if not isinstance(gold_span_str, str):
                logger.error(f"Example {i} has non-string target: {gold_span_str}. Skipping.")
                continue # Skip if target isn't a string

            try:
                with torch.no_grad():
                    # Generate output using greedy search
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=False # Force greedy
                    )

                # Decode only the generated part
                generated_text = self.tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

                # Parse predicted and gold spans
                pred_spans = parse_span_from_string(generated_text)
                gold_spans = parse_span_from_string(gold_span_str)

                tp += len(pred_spans & gold_spans)
                fp += len(pred_spans - gold_spans)
                fn += len(gold_spans - pred_spans)

            except Exception as e:
                 logger.error(f"Error during generation or parsing for example {i}: {e}", exc_info=True)
                 decode_errors += 1
                 # Penalize F1 by counting all gold spans as false negatives for this example
                 try:
                     gold_spans = parse_span_from_string(gold_span_str)
                     fn += len(gold_spans)
                 except Exception as parse_e:
                     logger.error(f"Could not parse gold spans for error penalty: {parse_e}")


        # Calculate Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info(f"Generative Evaluation Complete: TP={tp}, FP={fp}, FN={fn}, Decode Errors={decode_errors}/{total_examples}")
        logger.info(f"Span Metrics: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

         # Restore model training state
        if is_training:
            model.train()

        return {
            f"{self.metric_key_prefix}_ner_precision": precision,
            f"{self.metric_key_prefix}_ner_recall": recall,
            f"{self.metric_key_prefix}_ner_f1": f1,
            f"{self.metric_key_prefix}_decode_errors": decode_errors,
        }

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        logger.info(f"Callback evaluating at step {state.global_step}...")
        metrics = {}

        eval_dataloader = kwargs.get("eval_dataloader", None)
        if not eval_dataloader:
             logger.warning("Eval dataloader not found in kwargs. Cannot compute metrics.")
             return control

        # --- Fast Token Metrics (if enabled) ---
        # Requires model forward pass to have returned token_logits
        # Typically run using trainer.predict() which captures outputs
        # If using trainer.evaluate(), need custom compute_metrics in Trainer
        # Assuming we'll use trainer.predict() or adapt compute_metrics later
        # For now, focus on the generative eval triggered explicitly

        # --- Generative Span Metrics (if enabled) ---
        if self.gen_eval_enabled and model is not None:
            # Check if generation is currently enabled in Trainer args
            # The spec wants predict_with_generate=False during most steps,
            # but True for end-of-epoch/final eval.
            # We compute it here regardless, assuming this callback instance
            # is used for the specific generative eval runs.
            logger.info("Computing generative span metrics...")
            # Important: Need access to the *raw* dataset for gold span strings
            # The dataloader yields tensors. Need the original dataset used by the loader.
            # This assumes self.eval_dataset is the correct *raw* dataset instance.
            gen_metrics = self._compute_generative_metrics(model, self.eval_dataset)
            metrics.update(gen_metrics)
        elif not model:
            logger.warning("Model not provided to on_evaluate, cannot compute generative metrics.")


        # Log metrics (Trainer usually handles logging metrics returned by compute_metrics)
        # This callback primarily *computes* them; logging is handled by Trainer integration
        # We can log directly here if needed, e.g., to W&B if WandbCallback isn't sufficient
        if state.is_world_process_zero: # Log only on main process
             logger.info(f"Metrics computed at step {state.global_step}: {metrics}")
             # If using W&B explicitly:
             # import wandb
             # if wandb.run:
             #     wandb.log(metrics, step=state.global_step)

        # Storing metrics in state for Trainer to pick up (if needed)
        # state.log_history[-1].update(metrics)

        return control

    # We might need a custom compute_metrics function for the Trainer
    # that uses this callback's methods, especially for the fast token metrics
    # if predict_with_generate is False.

    def compute_metrics_for_trainer(self, eval_preds):
        """Method to be passed as `compute_metrics` to the Trainer.

        Args:
            eval_preds: An EvalPrediction object containing predictions and label_ids.
                        Predictions might be logits or generated IDs depending on
                        `predict_with_generate`.
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        predictions, label_ids = eval_preds

        # --- Fast Token Metrics --- #
        if self.fast_metrics_enabled:
            # Assumes predictions are token logits when predict_with_generate=False
            # The token_logits might be nested if multiple outputs were returned by model
            # Need to confirm the structure passed by Trainer

            token_logits = predictions # Default assumption
            token_labels = label_ids.get("token_labels", None) # Assuming labels passed as dict

            # Heuristic: Check if predictions look like logits (more than 1 dim, float type)
            # This check is weak and might need adjustment based on Trainer behavior
            is_logits = isinstance(predictions, np.ndarray) and predictions.ndim > 1 and np.issubdtype(predictions.dtype, np.floating)

            if is_logits and token_labels is not None:
                logger.info("Computing fast token metrics (seqeval) from logits...")
                # Get actual predictions
                token_preds = np.argmax(token_logits, axis=-1)
                # Ensure token_labels has the same shape for comparison
                if token_preds.shape == token_labels.shape:
                     token_metrics = self._compute_token_metrics(token_preds, token_labels)
                     metrics.update(token_metrics)
                else:
                     logger.warning(f"Shape mismatch between token predictions {token_preds.shape} and labels {token_labels.shape}. Skipping token metrics.")
            elif not is_logits:
                 logger.info("Predictions do not appear to be token logits. Skipping fast token metrics.")
            elif token_labels is None:
                 logger.info("Token labels not found in label_ids. Skipping fast token metrics.")

        # --- Generative Metrics --- #
        # These are harder to compute within compute_metrics if predict_with_generate=True
        # because we only get the generated IDs, not the original input or target strings.
        # The spec suggests running generate explicitly in the main script for these.
        # We will compute the primary `eval_ner_f1` here based on the assumption
        # that a separate mechanism (like the on_evaluate hook or main script logic)
        # has already computed and stored the generative F1, maybe in a shared object
        # or by logging it directly.
        # For simplicity, the Trainer will rely on the `metric_for_best_model` being
        # logged externally by the callback or main script logic when generation runs.
        # We return an empty dict here for the generative part, assuming the callback handles logging.

        # Placeholder for the primary metric required by load_best_model_at_end
        # This value MUST be logged by the generative eval process somehow.
        if f"{self.metric_key_prefix}_ner_f1" not in metrics:
             metrics[f"{self.metric_key_prefix}_ner_f1"] = 0.0 # Default if not computed/logged elsewhere

        logger.debug(f"compute_metrics returning: {metrics}")
        return metrics 