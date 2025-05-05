# Placeholder for NER Span Datamodule 

import os
import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl
import logging
from typing import Optional, Dict, List, Tuple, Set

logger = logging.getLogger(__name__)

def parse_span(txt: str) -> Set[Tuple[int, int]]:
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
            logger.warning(f"Could not parse span part: '{part}' in '{txt}'")
            pass
    return spans


class NERSpanDataset(Dataset):
    """Dataset for NER Span Generation task."""
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_input_len: int = 256, max_target_len: int = 64, generate_bio_labels: bool = False):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.generate_bio_labels = generate_bio_labels
        self.bio_map = {"O": 0, "B": 1, "I": 2} # O, B-span, I-span

        logger.info(f"Loading data from: {data_path}")
        try:
            self.data = datasets.load_dataset("json", data_files=data_path, split="train", streaming=False) # Stream later if needed
            logger.info(f"Loaded {len(self.data):,} examples.")
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            raise

    def __len__(self):
        return len(self.data)

    def _get_token_bio_labels(self, text_input: str, char_spans: Set[Tuple[int, int]], encoding) -> List[int]:
        """Converts character spans to token-level BIO labels."""
        # Initialize labels to 'O'
        token_labels = [self.bio_map["O"]] * len(encoding.input_ids)
        word_ids = encoding.word_ids()

        # Mark special tokens with -100
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                token_labels[token_idx] = -100 # Ignore loss for special tokens

        if not char_spans:
            return token_labels # All 'O' if no spans

        # Assign B/I tags based on character spans
        for start_char, end_char in char_spans:
            span_started = False
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue # Skip special tokens

                # Get character span for the current token
                token_char_span = encoding.word_to_chars(word_idx)
                if token_char_span is None: continue # Should not happen for non-special tokens

                tok_start, tok_end = token_char_span.start, token_char_span.end

                # Check for overlap or inclusion
                if tok_start >= start_char and tok_end <= end_char: # Token within span
                    if not span_started:
                        token_labels[token_idx] = self.bio_map["B"]
                        span_started = True
                    else:
                        token_labels[token_idx] = self.bio_map["I"]
                # Handle cases where a span might start/end mid-token (less common with wordpiece/bpe)
                # This basic logic tags any token overlapping with the span.
                # A more refined approach might be needed depending on tokenizer specifics.

        return token_labels


    def __getitem__(self, idx):
        item = self.data[idx]
        text_input = item['input'] # e.g., "<NER> Aspirin can trigger..."
        target_span_str = item['target'] # e.g., "<TAG> 0-7 ; 19-32"

        # Tokenize input
        encoding = self.tokenizer(
            text_input,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target span string for causal LM loss
        target_encoding = self.tokenizer(
            target_span_str,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Input IDs become labels for causal LM, padded tokens are -100
        labels = target_encoding.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        token_bio_labels = None
        if self.generate_bio_labels:
            # Requires original text *without* <NER> prefix for char alignment
            original_text = text_input.replace("<NER>", "").strip()
            char_spans = parse_span(target_span_str)
            # We need to tokenize the *original* text to get correct word_ids for char spans
            original_text_encoding = self.tokenizer(
                original_text,
                max_length=self.max_input_len - 1, # Account for removed <NER>
                truncation=True,
                # No padding needed here, just for word_ids
            )
            # Now, map the BIO labels based on original text encoding's word_ids
            # This part is tricky because the main input includes <NER>.
            # For simplicity, we'll generate BIO labels based *only* on the original text part,
            # and assume the model learns to ignore the <NER> token for this head.
            # A more robust solution might involve adjusting indices or masking.

            # Recalculate BIO labels based on the main input encoding, aligning best effort
            token_bio_labels = self._get_token_bio_labels(original_text, char_spans, encoding) # Pass main encoding
            # Pad token_bio_labels to max_input_len with -100
            padded_token_bio_labels = token_bio_labels + [-100] * (self.max_input_len - len(token_bio_labels))
            token_bio_labels = torch.tensor(padded_token_bio_labels[:self.max_input_len], dtype=torch.long)


        return {
            "input_ids": encoding.input_ids.squeeze(0), # Remove batch dim
            "attention_mask": encoding.attention_mask.squeeze(0),
            "labels": labels.squeeze(0), # For Causal LM loss
            "token_labels": token_bio_labels # Optional BIO ints for TC head
        }


class NERSpanDataModule(pl.LightningDataModule):
    def __init__(self,
                 tokenizer_path: str = "taskmerge/tokenizers/llama3_biomerge_tok",
                 data_dir: str = "taskmerge/data/union_span/ner",
                 max_input_len: int = 256,
                 max_target_len: int = 64,
                 train_fraction: float = 1.0,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 generate_bio_labels: bool = False, # Corresponds to fast_metrics
                 ):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.data_dir = data_dir
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.generate_bio_labels = generate_bio_labels
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Save hyperparameters for easy access
        self.save_hyperparameters()

    def prepare_data(self):
        # Download data, tokenizers, etc. Run once across processes.
        # In this case, data is assumed local, tokenizer might be downloaded if not present
        AutoTokenizer.from_pretrained(self.tokenizer_path)
        # Verify data exists
        if not os.path.exists(os.path.join(self.data_dir, "train.jsonl")):
             raise FileNotFoundError(f"Training data not found at {os.path.join(self.data_dir, 'train.jsonl')}")


    def setup(self, stage: Optional[str] = None):
        # Load tokenizer here for multi-GPU safety
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.tokenizer.pad_token is None:
            logger.warning("Tokenizer has no pad token; setting to EOS token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if stage == "fit" or stage is None:
            full_train_path = os.path.join(self.data_dir, "train.jsonl")
            full_train_dataset = NERSpanDataset(full_train_path, self.tokenizer, self.max_input_len, self.max_target_len, self.generate_bio_labels)

            if self.train_fraction < 1.0:
                num_samples = int(len(full_train_dataset) * self.train_fraction)
                logger.info(f"Using {self.train_fraction*100:.1f}% ({num_samples}) of training data.")
                # This basic subsetting might not be ideal for reproducibility across runs/nodes.
                # Consider using dataset.select() with a fixed seed if needed.
                indices = list(range(num_samples))
                self.train_dataset = torch.utils.data.Subset(full_train_dataset, indices)
            else:
                self.train_dataset = full_train_dataset

            val_path = os.path.join(self.data_dir, "dev.jsonl")
            self.val_dataset = NERSpanDataset(val_path, self.tokenizer, self.max_input_len, self.max_target_len, self.generate_bio_labels)
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            test_path = os.path.join(self.data_dir, "test.jsonl")
            self.test_dataset = NERSpanDataset(test_path, self.tokenizer, self.max_input_len, self.max_target_len, self.generate_bio_labels)
            logger.info(f"Test dataset size: {len(self.test_dataset)}")

        if stage == "predict":
             # Optionally handle predict stage setup if needed
             pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

# Example usage (for testing within the file)
if __name__ == '__main__':
    # Assume tokenizer is prepared and data exists
    # 1. Prepare tokenizer (run scripts/prepare_tokenizer.py first)
    # 2. Ensure data exists at taskmerge/data/union_span/ner/{train,dev,test}.jsonl
    #    Create dummy data for testing if needed.
    # Example: echo '{"input": "<NER> Sample text here.", "target": "<TAG> 7-11"}' > taskmerge/data/union_span/ner/train.jsonl
    #          echo '{"input": "<NER> Dev text here.", "target": "<TAG> 4-8"}' > taskmerge/data/union_span/ner/dev.jsonl
    #          echo '{"input": "<NER> Test text here.", "target": "<TAG> 5-9"}' > taskmerge/data/union_span/ner/test.jsonl

    import torch
    print("Running datamodule example...")

    # Make sure the paths are correct relative to where you run this
    tokenizer_path = "../../tokenizers/llama3_biomerge_tok" # Adjust path if running from src/datamodules
    data_dir = "../../data/union_span/ner"              # Adjust path

    # Check if dummy data exists, create if not
    os.makedirs(data_dir, exist_ok=True)
    for split in ["train", "dev", "test"]:
        fpath = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(fpath):
            print(f"Creating dummy data file: {fpath}")
            with open(fpath, "w") as f:
                if split == "train":
                    f.write('{"input": "<NER> Sample training text here.", "target": "<TAG> 7-11 ; 17-21"}\n')
                    f.write('{"input": "<NER> Another sentence.", "target": "<TAG>"}\n') # Example with no spans
                elif split == "dev":
                    f.write('{"input": "<NER> Dev text example.", "target": "<TAG> 4-8"}\n')
                else: # test
                    f.write('{"input": "<NER> Test text example.", "target": "<TAG> 5-9"}\n')

    dm = NERSpanDataModule(
        tokenizer_path=tokenizer_path,
        data_dir=data_dir,
        batch_size=2,
        generate_bio_labels=True, # Test BIO label generation
        num_workers=0 # Easier debugging
    )

    dm.prepare_data()
    dm.setup(stage='fit')

    print("\nTrain Dataloader Example Batch:")
    train_loader = dm.train_dataloader()
    try:
        batch = next(iter(train_loader))
        print("Keys:", batch.keys())
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention Mask shape:", batch['attention_mask'].shape)
        print("Labels shape:", batch['labels'].shape)
        print("Token Labels shape:", batch['token_labels'].shape) # Check if BIO labels are generated

        print("\nInput IDs (first example):")
        print(batch['input_ids'][0])
        print("Decoded Input:")
        print(dm.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))

        print("\nLabels (first example):")
        print(batch['labels'][0])
        # Replace -100 with pad_token_id for decoding labels
        decodable_labels = batch['labels'][0].clone()
        decodable_labels[decodable_labels == -100] = dm.tokenizer.pad_token_id
        print("Decoded Labels:")
        print(dm.tokenizer.decode(decodable_labels, skip_special_tokens=False))


        print("\nToken Labels (first example, BIO):")
        print(batch['token_labels'][0])
        # Map back to O, B, I, <PAD>
        id_to_bio = {v:k for k, v in dm.train_dataset.dataset.bio_map.items()} # Access base dataset if subset
        id_to_bio[-100] = "<PAD>"
        bio_str = [id_to_bio[label_id.item()] for label_id in batch['token_labels'][0]]
        print(bio_str)

        # Combine tokens and BIO labels for inspection
        tokens = dm.tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
        print("\nTokens with BIO labels:")
        for token, bio in zip(tokens, bio_str):
            print(f"{token:<20} {bio}")


    except StopIteration:
        print("Could not get batch from train_loader. Is the data file empty or corrupt?")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\nDatamodule example finished.") 