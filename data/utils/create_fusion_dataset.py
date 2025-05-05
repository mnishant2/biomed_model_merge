#!/usr/bin/env python
"""
Script to create a small sample dataset for adapter fusion.
It extracts a balanced sample (totaling 5% of combined data) from NER and EL datasets,
interleaves them, and ensures consistency by removing these samples from all other datasets.
"""
import os
import json
import random
import hashlib
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path):
    """Load data from a jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    """Save data to a jsonl file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved {len(data)} examples to {file_path}")

def hash_example(example):
    """Create a unique hash for an example based on input text."""
    # Create a hash based on the input text to identify examples
    input_text = example.get('input', '')
    return hashlib.md5(input_text.encode('utf-8')).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Create adapter fusion dataset")
    parser.add_argument('--data_root', type=str, default='data/union',
                        help='Root directory containing all datasets')
    parser.add_argument('--sample_percent', type=float, default=5.0,
                        help='Percentage of TOTAL data to sample (default: 5.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    data_root = Path(args.data_root)
    
    # Define directories to process
    all_dirs = [
        'NER', 
        'EL', 
        'joint_decode_EL', 
        'fullft', 
        'head_concat', 
        'SAPBERT', 
        'BioBERT'
    ]
    
    # Create output directory for fusion data
    fusion_dir = data_root / 'fusion'
    os.makedirs(fusion_dir, exist_ok=True)
    
    # Load NER and EL datasets with correct file names
    ner_file = data_root / 'NER' / 'union_train_ner.jsonl'
    el_file = data_root / 'EL' / 'union_train_el.jsonl'
    
    logger.info(f"Loading NER data from {ner_file}")
    ner_data = load_jsonl(ner_file)
    logger.info(f"Loaded {len(ner_data)} NER examples")
    
    logger.info(f"Loading EL data from {el_file}")
    el_data = load_jsonl(el_file)
    logger.info(f"Loaded {len(el_data)} EL examples")
    
    # Calculate total dataset size and target sample size (5% of total)
    total_data_size = len(ner_data) + len(el_data)
    total_sample_size = max(2, int(total_data_size * args.sample_percent / 100))
    
    # Calculate balanced sample sizes (approximately half of total sample size for each dataset)
    # Adjust if datasets are very imbalanced to ensure proper representation
    ner_sample_size = max(1, int(total_sample_size * 0.5))
    el_sample_size = max(1, total_sample_size - ner_sample_size)
    
    # Log data sizes and sampling information
    logger.info(f"Total data size: {total_data_size} examples")
    logger.info(f"Target sample size (total): {total_sample_size} examples ({args.sample_percent}% of total)")
    logger.info(f"Sampling {ner_sample_size} NER examples (~{(ner_sample_size/len(ner_data)*100):.2f}% of NER data)")
    logger.info(f"Sampling {el_sample_size} EL examples (~{(el_sample_size/len(el_data)*100):.2f}% of EL data)")
    
    # Create random samples
    ner_sample = random.sample(ner_data, ner_sample_size)
    el_sample = random.sample(el_data, el_sample_size)
    
    # Add dataset identifiers for balancing during training
    for item in ner_sample:
        item['dataset'] = 'NER'
    
    for item in el_sample:
        item['dataset'] = 'EL'
    
    # Interleave samples
    fusion_data = []
    for i in range(max(len(ner_sample), len(el_sample))):
        if i < len(ner_sample):
            fusion_data.append(ner_sample[i])
        if i < len(el_sample):
            fusion_data.append(el_sample[i])
    
    # Shuffle the interleaved data
    random.shuffle(fusion_data)
    
    # Save fusion dataset
    fusion_file = fusion_dir / 'dev_mixed.jsonl'
    save_jsonl(fusion_data, fusion_file)
    logger.info(f"Created fusion dataset with {len(fusion_data)} examples")
    
    # Create a set of hashes for the sampled examples
    sampled_hashes = set(hash_example(ex) for ex in fusion_data)
    
    # Remove sampled examples from all training sets
    logger.info("Removing sampled examples from all other datasets")
    for dir_name in tqdm(all_dirs):
        if dir_name == 'NER':
            train_file = data_root / dir_name / 'union_train_ner.jsonl'
        elif dir_name == 'EL':
            train_file = data_root / dir_name / 'union_train_el.jsonl'
        else:
            # For other directories, try to find the appropriate train file
            potential_files = list((data_root / dir_name).glob('*train*.jsonl'))
            if potential_files:
                train_file = potential_files[0]
            else:
                logger.warning(f"No train file found in {dir_name}, skipping")
                continue
                
        if not train_file.exists():
            logger.warning(f"File {train_file} not found, skipping")
            continue
        
        # Load data
        try:
            train_data = load_jsonl(train_file)
            initial_count = len(train_data)
            
            # Filter out the sampled examples
            filtered_data = [ex for ex in train_data if hash_example(ex) not in sampled_hashes]
            
            # Save filtered data
            save_jsonl(filtered_data, train_file)
            
            removed = initial_count - len(filtered_data)
            logger.info(f"Removed {removed} examples from {train_file}")
        except Exception as e:
            logger.error(f"Error processing {train_file}: {e}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 