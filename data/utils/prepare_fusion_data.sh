#!/bin/bash
# Script to create a fusion dataset for adapter fusion training

set -e  # Exit on any error

# Parse command line arguments
SAMPLE_PERCENT=5.0
DATA_ROOT="data/union"
SEED=42

function print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --percent VALUE    Percentage of TOTAL data to sample (default: 5.0)"
    echo "  --data_root PATH   Path to data root directory (default: data/union)"
    echo "  --seed VALUE       Random seed for reproducibility (default: 42)"
    echo "  --help             Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --percent)
            SAMPLE_PERCENT="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

echo "Creating fusion dataset with ${SAMPLE_PERCENT}% of TOTAL data (balanced from NER and EL)"
echo "Data root: ${DATA_ROOT}"
echo "Random seed: ${SEED}"

# Make sure directories exist
mkdir -p "${DATA_ROOT}/fusion"

# Count original dataset sizes before modification
NER_ORIG=$(wc -l < "${DATA_ROOT}/NER/union_train_ner.jsonl" 2>/dev/null)
EL_ORIG=$(wc -l < "${DATA_ROOT}/EL/union_train_el.jsonl" 2>/dev/null)
TOTAL_ORIG=$((NER_ORIG + EL_ORIG))

# Run the python script with miniconda Python
~/miniconda/bin/python3 data/utils/create_fusion_dataset.py \
    --data_root "${DATA_ROOT}" \
    --sample_percent "${SAMPLE_PERCENT}" \
    --seed "${SEED}"

# Count new dataset sizes after modification
NER_NEW=$(wc -l < "${DATA_ROOT}/NER/union_train_ner.jsonl" 2>/dev/null)
EL_NEW=$(wc -l < "${DATA_ROOT}/EL/union_train_el.jsonl" 2>/dev/null)
FUSION_SIZE=$(wc -l < "${DATA_ROOT}/fusion/dev_mixed.jsonl" 2>/dev/null)

echo "Fusion dataset creation complete!"
echo "The fusion dataset is saved at: ${DATA_ROOT}/fusion/dev_mixed.jsonl"

# Print statistics about the datasets
echo ""
echo "Dataset statistics:"
echo "==================="
echo "Original NER dataset size: ${NER_ORIG}"
echo "Original EL dataset size: ${EL_ORIG}"
echo "Original total dataset size: ${TOTAL_ORIG}"
echo ""
echo "NER samples extracted: $((NER_ORIG - NER_NEW))"
echo "EL samples extracted: $((EL_ORIG - EL_NEW))"
echo "Fusion dataset size: ${FUSION_SIZE}"
echo "Percentage of total: $(echo "scale=2; 100*${FUSION_SIZE}/${TOTAL_ORIG}" | bc)%" 