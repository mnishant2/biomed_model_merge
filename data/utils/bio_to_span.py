# bio_to_span_converter.py
# ─────────────────────────────────────────────────────────────────────────────
"""Utility script that rewrites the *NER* JSONL files containing BIO‐tag 
sequences into the compact *span‑string* format discussed in the final design.

Usage (CLI):
    python bio_to_span_converter.py \
        --input data/union/ner/train.jsonl \
        --output data/union_span/ner/train.jsonl \
        --tokenizer aaditya/Llama3-OpenBioLLM-8B \
        --strip-prefix 1

Notes
-----
* The script assumes the JSONL lines follow the structure produced by the
  previous pipeline:
      {"input": "<NER> sentence …", "target": "B-ENT O O …", ...}
* It produces new lines with:
      {"input": "<NER> sentence …", "target": "<TAG> 0-7 ; 19-32", ...}
  preserving all other metadata.
"""

import json
from pathlib import Path
from typing import List, Tuple
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

def bio_tags_to_char_spans(sentence: str,
                           bio_tags: List[str],
                           tokenizer,
                           strip_prefix: int = 0) -> List[Tuple[int, int]]:
    """Convert BIO tags aligned to *tokenised* sentence → char spans."""
    enc = tokenizer(sentence,
                    add_special_tokens=False,
                    return_offsets_mapping=True)
    offsets = enc["offset_mapping"][strip_prefix:]
    tags    = bio_tags[strip_prefix:]

    spans = []
    i = 0
    while i < len(tags):
        if tags[i].startswith("B"):
            start_char = offsets[i][0]
            end_char   = offsets[i][1]
            j = i + 1
            while j < len(tags) and tags[j].startswith("I"):
                end_char = offsets[j][1]
                j += 1
            spans.append((start_char, end_char))
            i = j
        else:
            i += 1
    return spans

def spans_to_string(spans: List[Tuple[int, int]]) -> str:
    """Serialise list of (start,end) to "0-7 ; 19-32"."""
    return " ; ".join([f"{s}-{e}" for s, e in spans])

def convert_file(in_path: Path, out_path: Path, tokenizer, strip_prefix: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in tqdm(fin, desc=f"{in_path.name}"):
            ex = json.loads(line)
            sentence = ex["input"].split(" ", strip_prefix)[-1] if strip_prefix else ex["input"]
            bio_tags = ex["target"].split()
            spans = bio_tags_to_char_spans(sentence, bio_tags, tokenizer, strip_prefix)
            span_str = spans_to_string(spans)
            ex["target"] = f"<TAG> {span_str}" if span_str else "<TAG>"
            fout.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to BIO jsonl")
    parser.add_argument("--output", required=True, help="where to write span jsonl")
    parser.add_argument("--tokenizer", default="aaditya/Llama3-OpenBioLLM-8B")
    parser.add_argument("--strip-prefix", type=int, default=0,
                        help="#tokens to skip before BIO seq (e.g. 1 for <NER>")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    convert_file(Path(args.input), Path(args.output), tok, args.strip_prefix)

    print("✓ Conversion complete →", args.output)
