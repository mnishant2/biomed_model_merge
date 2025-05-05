#!/usr/bin/env python
"""
Build FAISS indices from SapBERT embeddings for fast entity retrieval.
"""
import os
import argparse
import json
import numpy as np
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS indices from SapBERT embeddings")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the SapBERT model directory",
    )
    parser.add_argument(
        "--concept-file",
        type=str,
        default="data/union/SAPBERT/concepts.jsonl",
        help="JSONL file containing concept data (id, name, aliases)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/indices",
        help="Directory to save the FAISS index and metadata",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding entities",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="IndexFlatIP",
        choices=["IndexFlatIP", "IndexIVFFlat", "IndexHNSW"],
        help="Type of FAISS index to build",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load SapBERT model
    print(f"Loading SapBERT model from {args.model_path}")
    model = SentenceTransformer(args.model_path)
    model.eval()
    
    # Load concept data
    print(f"Loading concept data from {args.concept_file}")
    concepts = []
    with open(args.concept_file, "r") as f:
        for line in f:
            concepts.append(json.loads(line))
    
    print(f"Loaded {len(concepts)} concepts")
    
    # Create embeddings for concept names
    print("Generating embeddings for concept names")
    concept_names = [c["name"] for c in concepts]
    concept_ids = [c["id"] for c in concepts]
    
    # Encode in batches to avoid OOM
    embeddings = []
    for i in tqdm(range(0, len(concept_names), args.batch_size)):
        batch = concept_names[i:i + args.batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    print(f"Building {args.index_type} index")
    dimension = embeddings.shape[1]
    
    if args.index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dimension)
    elif args.index_type == "IndexIVFFlat":
        # For IVF index, we need a quantizer
        quantizer = faiss.IndexFlatIP(dimension)
        nlist = min(4096, 4 * int(np.sqrt(len(concept_names))))
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        # Train on a subset of vectors
        train_size = min(100000, len(embeddings))
        index.train(embeddings[:train_size])
    elif args.index_type == "IndexHNSW":
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    
    # Add vectors to the index
    print("Adding vectors to index")
    index.add(embeddings)
    
    # Save index and metadata
    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, f"sapbert_{args.index_type}.index")
    meta_path = os.path.join(args.output_dir, "concept_meta.json")
    
    print(f"Saving index to {index_path}")
    faiss.write_index(index, index_path)
    
    print(f"Saving concept metadata to {meta_path}")
    with open(meta_path, "w") as f:
        json.dump({"ids": concept_ids, "names": concept_names}, f)
    
    print("Done!")

if __name__ == "__main__":
    main() 