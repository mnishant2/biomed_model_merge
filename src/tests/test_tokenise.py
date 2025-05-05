# Placeholder for Datamodule Tokenization Tests 

import pytest
import torch
import os
from transformers import AutoTokenizer
from src.datamodules.ner_span import NERSpanDataModule, NERSpanDataset # Adjust import path if necessary
import json
import logging

logger = logging.getLogger(__name__)

# Sample data (first 4 lines from sample_span.jsonl)
dummy_data = [
    json.loads('{"task": "NER", "input": "<NER> CAD 3)severe mitral regurgitation 4)Atrial fibrillation, s/p ICD-not anticoagulated, on amio 5)Peripheral vascular disease, s/p bypass leg surgery 6)ESRD on HD with R tunneled line 7)Anemia on Procrit and iron supplementation 8)? CVA years ago with left facial numbness 9)Hypothyroidism 10) s/p right above the knee popliteal bypass graft in and a left femoral popliteal artery bypass graft with revision that included the left femoral to anterior tibial artery jump graft in 11) BRBPR: hospitalized , EGD showed severe gastritis and colonoscopy showed numerous diverticuli and adenomatous polyps, tagged RBC scan negative 12) Cdiff 13) L foot hematoma requiring I and D hospitalization Social History: Smoked 1 ppd x 50 yrs, quit 1998. Reported heavy EtOH use in past, none currently. currently at . Lives with his wife in . 2 children living in .Retired maintenance worker at . Family History: NC Physical Exam: VS: 97.9 125/59 82 RR", "target": "<TAG> 12-33 ; 36-55 ; 95-122 ; 149-153 ; 183-189 ; 229-233 ; 248-269 ; 272-286 ; 479-485 ; 519-529 ; 552-573 ; 577-596 ; 626-632 ; 643-652", "doc_id": "share_clef_21413-012450-discharge_summary", "dataset": "share_clef"}'),
    json.loads('{"task": "NER", "input": "<NER>  jump graft in 11) BRBPR: hospitalized , EGD showed severe gastritis and colonoscopy showed numerous diverticuli and adenomatous polyps, tagged RBC scan negative 12) Cdiff 13) L foot hematoma requiring I and D hospitalization Social History: Smoked 1 ppd x 50 yrs, quit 1998. Reported heavy EtOH use in past, none currently. currently at . Lives with his wife in . 2 children living in .Retired maintenance worker at . Family History: NC Physical Exam: VS: 97.9 125/59 82 RR: 11 98% on 2L Gen: elderly man lying in bed, dozing, in no apparent distress HEENT: PERRLA, neck supple Cardiac: irregular, murmur Pulm: scattered wheezes in all lungs fields Abd: soft, NT, ND, +BS Ext: + 2 edema, + 2 DP pulses Neuro: alert, oriented, variably cooperative with exam, CN intact, strength 3-4/5 in BL upper and R lower extremity, (inconsistant) LLE seems to be 1-2/5, cerbellar exam", "target": "<TAG> 18-24 ; 58-68 ; 91-112 ; 116-135 ; 165-171 ; 182-191 ; 533-551 ; 587-597 ; 598-605 ; 621-629 ; 649-653 ; 660-663 ; 664-667 ; 681-687", "doc_id": "share_clef_21413-012450-discharge_summary", "dataset": "share_clef"}'),
    json.loads('{"task": "NER", "input": "<NER> : 11 98% on 2L Gen: elderly man lying in bed, dozing, in no apparent distress HEENT: PERRLA, neck supple Cardiac: irregular, murmur Pulm: scattered wheezes in all lungs fields Abd: soft, NT, ND, +BS Ext: + 2 edema, + 2 DP pulses Neuro: alert, oriented, variably cooperative with exam, CN intact, strength 3-4/5 in BL upper and R lower extremity, (inconsistant) LLE seems to be 1-2/5, cerbellar exam wnl, reflexes +3 in RLE, and +2 elsewhere. Sensation diminished in lower extremities to shins BL. Pertinent Results: CT abd/pelvis: 1. Ultrasound and CT findings are consistent with adenomyomatosis of the gallbladder. There is also a gallstone present. 2. Mass lesion within the lower lobe of the left lung, measuring up to 2.7 cm in diameter. Correlation with outside imaging studies would be helpful. 3. Small bilateral pleural effusions. 4. Massive splenomegaly, with a wedge shaped", "target": "<TAG> 59-77 ; 113-123 ; 124-131 ; 147-155 ; 175-179 ; 186-189 ; 190-193 ; 207-213 ; 441-451 ; 580-596 ; 603-615 ; 632-642 ; 654-666 ; 810-838 ; 850-863", "doc_id": "share_clef_21413-012450-discharge_summary", "dataset": "share_clef"}'),
    json.loads('{"task": "NER", "input": "<NER>  wnl, reflexes +3 in RLE, and +2 elsewhere. Sensation diminished in lower extremities to shins BL. Pertinent Results: CT abd/pelvis: 1. Ultrasound and CT findings are consistent with adenomyomatosis of the gallbladder. There is also a gallstone present. 2. Mass lesion within the lower lobe of the left lung, measuring up to 2.7 cm in diameter. Correlation with outside imaging studies would be helpful. 3. Small bilateral pleural effusions. 4. Massive splenomegaly, with a wedge shaped peripheral hypodensity concerning for infarct. 5. Atherosclerotic disease of the abdominal aorta with aortobifemoral bypass. 6. Tiny hypodensities of the liver and kidneys, too small to characterize. 7. Sigmoid diverticulosis, without evidence of diverticulitis. RUQ US: 1. Collapsed gallbladder with calcifications of the wall, which is thickened. There is no ultrasound evidence of acute cholecystitis. 2. Hemangioma of the left lobe of the liver. 3. No intra- or", "target": "<TAG> 43-53 ; 182-198 ; 205-217 ; 234-244 ; 256-268 ; 412-440 ; 452-465 ; 486-509 ; 524-532 ; 536-560 ; 619-633 ; 689-712 ; 733-748 ; 760-782 ; 787-802 ; 824-834 ; 870-890 ; 894-905", "doc_id": "share_clef_21413-012450-discharge_summary", "dataset": "share_clef"}')
]


# Path to the *actual* prepared tokenizer in the workspace
# Adjust this path if it's different in your setup
ACTUAL_TOKENIZER_PATH = "tokenizers/llama3_biomerge_tok"

# Fixture to create temporary data files using real samples
@pytest.fixture(scope="module")
def setup_real_data(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("real_data_test") / "data" / "union_span" / "ner"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Write the sample data to temporary files
    for split in ["train", "dev", "test"]:
        with open(data_dir / f"{split}.jsonl", "w") as f:
            for line in dummy_data: # Using the real samples loaded above
                f.write(json.dumps(line) + "\n")

    return str(data_dir)

# Fixture to provide the path to the actual tokenizer
@pytest.fixture(scope="module")
def actual_tokenizer_path():
    if not os.path.exists(ACTUAL_TOKENIZER_PATH):
        pytest.skip(f"Actual tokenizer not found at {ACTUAL_TOKENIZER_PATH}. Run prepare_tokenizer.py first.")
    return ACTUAL_TOKENIZER_PATH

# Test Datamodule Initialization with real tokenizer path
def test_datamodule_init_real_tok(actual_tokenizer_path, setup_real_data):
    data_dir = setup_real_data
    dm = NERSpanDataModule(
        tokenizer_path=actual_tokenizer_path,
        data_dir=data_dir,
        batch_size=2,
        generate_bio_labels=True,
        num_workers=0
    )
    assert dm.hparams.tokenizer_path == actual_tokenizer_path
    assert dm.hparams.batch_size == 2
    assert dm.hparams.generate_bio_labels is True
    assert dm.tokenizer is None # Tokenizer loaded in setup

# Test Datamodule Setup with real tokenizer and data
def test_datamodule_setup_real(actual_tokenizer_path, setup_real_data):
    data_dir = setup_real_data
    dm = NERSpanDataModule(tokenizer_path=actual_tokenizer_path, data_dir=data_dir, batch_size=1, generate_bio_labels=True, num_workers=0)
    dm.prepare_data() # Should check tokenizer exists
    dm.setup(stage="fit")
    assert dm.tokenizer is not None
    assert isinstance(dm.tokenizer, AutoTokenizer)
    assert "<NER>" in dm.tokenizer.vocab # Check special token
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert len(dm.train_dataset) == len(dummy_data)
    assert len(dm.val_dataset) == len(dummy_data)
    dm.setup(stage="test")
    assert dm.test_dataset is not None
    assert len(dm.test_dataset) == len(dummy_data)

# Test Dataloader Output Shape and Types with real data
def test_dataloader_output_real(actual_tokenizer_path, setup_real_data):
    data_dir = setup_real_data
    max_input_len = 256 # Use spec length
    max_target_len = 64  # Use spec length
    batch_size = 2
    dm = NERSpanDataModule(
        tokenizer_path=actual_tokenizer_path,
        data_dir=data_dir,
        batch_size=batch_size,
        max_input_len=max_input_len,
        max_target_len=max_target_len,
        generate_bio_labels=True,
        num_workers=0
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert isinstance(batch, dict)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert "token_labels" in batch

    assert batch["input_ids"].shape == (batch_size, max_input_len)
    assert batch["attention_mask"].shape == (batch_size, max_input_len)
    assert batch["labels"].shape == (batch_size, max_target_len)
    assert batch["token_labels"].shape == (batch_size, max_input_len)

    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["token_labels"].dtype == torch.long

# Test Label Padding (-100) with real data
def test_label_padding_real(actual_tokenizer_path, setup_real_data):
    data_dir = setup_real_data
    dm = NERSpanDataModule(tokenizer_path=actual_tokenizer_path, data_dir=data_dir, batch_size=1, max_target_len=64, num_workers=0)
    dm.prepare_data()
    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    labels = batch["labels"][0]
    tokenizer = dm.tokenizer
    # Get the first data item used in the fixture
    target_str = dummy_data[0]['target']
    target_tokens = tokenizer(target_str, max_length=dm.hparams.max_target_len, truncation=True)["input_ids"]

    # Check that padding in the labels tensor corresponds to -100
    assert (labels[len(target_tokens):] == -100).all()
    # Check that within the real tokens, there isn't -100
    if len(target_tokens) > 0 and len(target_tokens) < dm.hparams.max_target_len:
         assert not (labels[:len(target_tokens)] == -100).any()

# Test BIO Label Generation Logic with real data
def test_bio_label_generation_real(actual_tokenizer_path, setup_real_data):
    data_dir = setup_real_data
    dm = NERSpanDataModule(tokenizer_path=actual_tokenizer_path, data_dir=data_dir, batch_size=1, max_input_len=256, generate_bio_labels=True, num_workers=0)
    dm.prepare_data()
    dm.setup(stage="fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader)) # Get the first sample
    input_ids = batch["input_ids"][0]
    token_labels = batch["token_labels"][0]
    tokenizer = dm.tokenizer

    # Example 1 (dummy_data[0]): Has several spans
    active_labels = token_labels[token_labels != -100]
    assert (1 in active_labels) or (2 in active_labels) # Should contain B or I tags

    # We need an example with no entities to test the 'all O' case.
    # Modify fixture if necessary or skip this part if no such sample exists in the first 4 lines.
    # Assuming dummy_data[1] might have no *parseable* spans based on its target="<TAG>"
    # Let's load the second item specifically
    # Need to access the underlying dataset if using Subset
    if isinstance(dm.train_dataset, torch.utils.data.Subset):
        dataset_instance = dm.train_dataset.dataset
    else:
        dataset_instance = dm.train_dataset

    # Find an index with no spans if available in the small dataset
    no_span_idx = -1
    for i in range(len(dataset_instance)):
        if dataset_instance.data[i]['target'] == '<TAG>':
            no_span_idx = i
            break

    if no_span_idx != -1:
        item = dataset_instance[no_span_idx]
        # Collate manually for a single item test
        collated_batch = dm.collate_fn([item]) # Assuming default collate_fn is sufficient here
        token_labels_no_entity = collated_batch["token_labels"][0]
        active_labels_no_entity = token_labels_no_entity[token_labels_no_entity != -100]
        assert (active_labels_no_entity == 0).all() # Expect only O (0) tags
    else:
        logger.warning("Test data subset does not contain an example with no spans ('<TAG>'). Skipping 'all O' BIO check.")


# Test Train Fraction with real data
def test_train_fraction_real(actual_tokenizer_path, setup_real_data):
    data_dir = setup_real_data
    total_samples = len(dummy_data)
    # Use floor for fraction calculation based on subset logic
    expected_subset_size = int(total_samples * 0.5)

    dm = NERSpanDataModule(tokenizer_path=actual_tokenizer_path, data_dir=data_dir, train_fraction=0.5, num_workers=0)
    dm.prepare_data()
    dm.setup(stage="fit")
    assert len(dm.train_dataset) == expected_subset_size

    dm_full = NERSpanDataModule(tokenizer_path=actual_tokenizer_path, data_dir=data_dir, train_fraction=1.0, num_workers=0)
    dm_full.prepare_data()
    dm_full.setup(stage="fit")
    assert len(dm_full.train_dataset) == total_samples 