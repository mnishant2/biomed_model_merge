#!/usr/bin/env python
"""Offline span-F1 evaluation using a running vLLM OpenAI-compatible server.

Example Usage:

1. Launch Server (replace checkpoint path):
   python -m vllm.entrypoints.openai.api_server \
       --model /path/to/your/finetuned/checkpoint \
       --tokenizer taskmerge/tokenizers/llama3_biomerge_tok \
       --dtype bfloat16 \
       --port 8001 &

2. Run Evaluation:
   python taskmerge/src/scripts/eval_vllm.py \
       --split dev \
       --port 8001 \
       --data_dir taskmerge/data/union_span/ner
"""

import argparse
import requests
import datasets
from tqdm import tqdm
import logging
import re
from typing import Set, Tuple
import time
from http import HTTPStatus

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
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

def query_vllm_server(prompt: str, port: int, max_tokens: int = 64, retries: int = 3, delay: int = 5) -> str:
    """Sends a request to the vLLM OpenAI API server and returns the generated text."""
    api_url = f"http://localhost:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "model", # Placeholder model name, vLLM uses the one loaded
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0, # Greedy decoding
        "stop": ["<0xF0><0x9F><0xA7><0xBF>", "<|eot_id|>", tokenizer.eos_token if 'tokenizer' in globals() else None], # Add relevant stop tokens
    }
    payload = {k:v for k,v in payload.items() if v is not None} # remove None stop tokens

    for attempt in range(retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            response_json = response.json()
            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["text"]
            else:
                logger.warning(f"Unexpected response format: {response_json}")
                return "" # Return empty string on unexpected format
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.error("Max retries reached. Failing request.")
                return "" # Return empty string on failure
        except Exception as e:
            logger.error(f"An unexpected error occurred during API request: {e}")
            return "" # Return empty string on other errors

def main(split="dev", port=8001, data_dir="taskmerge/data/union_span/ner", max_new_tokens=64):
    logger.info(f"Starting vLLM evaluation on split '{split}'")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"vLLM server port: {port}")
    logger.info(f"Max new tokens for generation: {max_new_tokens}")

    # Load dataset (using datasets library)
    data_file = f"{data_dir}/{split}.jsonl"
    try:
        ds = datasets.load_dataset("json", data_files={"data": data_file})["data"]
        logger.info(f"Loaded {len(ds)} examples from {data_file}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {data_file}: {e}")
        return

    # Initialize metrics
    tp, fp, fn = 0, 0, 0
    total_examples = 0
    request_failures = 0

    # Process dataset
    for ex in tqdm(ds, desc=f"{split} eval"):
        total_examples += 1
        # Construct prompt as per model's training format
        prompt = ex["input"] + " <TAG>" # Add the trigger token
        gold_span_str = ex["target"]

        if not isinstance(gold_span_str, str):
            logger.warning(f"Example {total_examples-1} has non-string target: {gold_span_str}. Skipping.")
            continue

        # Query server
        generated_text = query_vllm_server(prompt, port, max_tokens=max_new_tokens)

        if generated_text:
            # Parse spans
            pred_spans = parse_span_from_string(generated_text)
            gold_spans = parse_span_from_string(gold_span_str)

            # Update counts
            tp += len(pred_spans & gold_spans)
            fp += len(pred_spans - gold_spans)
            fn += len(gold_spans - pred_spans)
        else:
            request_failures += 1
            # Penalize F1 if request failed by counting gold spans as FN
            try:
                gold_spans = parse_span_from_string(gold_span_str)
                fn += len(gold_spans)
            except Exception as parse_e:
                logger.error(f"Could not parse gold spans for error penalty after request failure: {parse_e}")

    # Calculate final metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logger.info("--- Evaluation Summary ---")
    logger.info(f"Processed examples: {total_examples}")
    logger.info(f"Request failures:   {request_failures}")
    logger.info(f"True Positives (TP): {tp}")
    logger.info(f"False Positives (FP):{fp}")
    logger.info(f"False Negatives (FN):{fn}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"Span F1:   {f1:.4f}")
    logger.info("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NER model via vLLM OpenAI server.")
    parser.add_argument("--split", default="dev", help="Dataset split to evaluate (e.g., dev, test)")
    parser.add_argument("--port", type=int, default=8001, help="Port where vLLM server is running")
    parser.add_argument("--data_dir", default="taskmerge/data/union_span/ner", help="Directory containing the .jsonl data files")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens for vLLM generation")
    # Need tokenizer for stop tokens
    try:
        from transformers import AutoTokenizer
        # Make sure this path is correct relative to where the script is run
        # Or pass it as an argument
        tokenizer = AutoTokenizer.from_pretrained("taskmerge/tokenizers/llama3_biomerge_tok")
    except ImportError:
        logger.error("Transformers library not found or tokenizer path is incorrect.")
        tokenizer = None
    except Exception as e:
        logger.error(f"Could not load tokenizer: {e}")
        tokenizer = None


    args = parser.parse_args()
    main(**vars(args)) 