# Placeholder for model forward pass tests 

import pytest
import torch
import os
import hydra
from omegaconf import OmegaConf

# Use fixtures from test_tokenise if applicable and adjust paths
# Assuming setup_real_data and actual_tokenizer_path fixtures exist
# Need to adjust relative paths for config loading
# This test assumes it runs from the project root where `pytest` is called.
CONF_PATH = "taskmerge/conf"

@pytest.fixture(scope="module")
def cfg():
    """Load Hydra config for testing."""
    # Compose config similar to how main.py would, but override for testing
    hydra.initialize(config_path=os.path.relpath(CONF_PATH, "."), version_base=None)
    cfg = hydra.compose(config_name="config", overrides=[
        "model=unsloth_lora_bf16",
        "task=ner_span",
        "runtime=a100", # Use a100 as base
        f"model.tokenizer_path=taskmerge/tokenizers/llama3_biomerge_tok", # Use relative path from root
        # Ensure model/runtime settings are minimal for CPU/memory if needed
        "model.lora_r=2", # Smaller LoRA rank for faster test
        "model.lora_alpha=4",
        "++runtime.training.bf16=false", # Disable bf16 if testing on CPU or non-bf16 GPU
        "++runtime.training.per_device_train_batch_size=1",
        "++runtime.training.gradient_accumulation_steps=1",
        "task.datamodule.num_workers=0",
        # Point data to a dummy path that the datamodule test setup creates
        # This requires the datamodule fixture to run first or data to exist
        # For simplicity, assume datamodule runs ok - focus on forward pass here
        # We won't actually load real data here, just test model forward
    ])
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    return cfg

@pytest.fixture(scope="module")
def initialized_model(cfg):
    """Initialize the model for testing."""
    # Skip if tokenizer doesn't exist
    tokenizer_path = cfg.model.tokenizer_path
    if not os.path.exists(tokenizer_path):
        pytest.skip(f"Tokenizer not found at {tokenizer_path}, cannot initialize model.")

    # Disable bf16 specifically for model init if test env doesn't support it
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        print("BF16 not supported or no CUDA, initializing model with default dtype (float32)")
        cfg.model.dtype_str = None
        cfg.model.load_in_4bit = False # Ensure 4bit is off too

    try:
        model = hydra.utils.instantiate(cfg.model)
        return model
    except Exception as e:
        pytest.fail(f"Failed to initialize model: {e}")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Forward test requires CUDA (can adapt for CPU)")
def test_forward_pass(initialized_model):
    """Test if the model forward pass runs and returns a loss tensor."""
    model = initialized_model
    tokenizer = model.tokenizer
    device = torch.device("cuda")
    model.to(device)
    model.train() # Ensure model is in training mode

    # Create dummy batch
    dummy_text = ["<NER> Sample one.", "<NER> Sample two with entity."]
    dummy_target = ["<TAG> 7-10", "<TAG> 20-26"]
    dummy_bio = [
        [0, 0, 1, 0, -100], # O O B O PAD
        [0, 0, 0, 0, 1, -100] # O O O O B PAD
    ]

    inputs = tokenizer(dummy_text, return_tensors="pt", padding="longest", truncation=True, max_length=16)
    targets = tokenizer(dummy_target, return_tensors="pt", padding="longest", truncation=True, max_length=8)
    labels = targets.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # Pad BIO labels manually to match input length
    max_len = inputs.input_ids.shape[1]
    token_labels_list = []
    for bio_seq in dummy_bio:
        padded_bio = bio_seq[:max_len] + [-100] * (max_len - len(bio_seq))
        token_labels_list.append(padded_bio)
    token_labels = torch.tensor(token_labels_list, dtype=torch.long)

    # Move batch to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels.to(device)
    token_labels = token_labels.to(device)

    try:
        outputs = model(**inputs, labels=labels, token_labels=token_labels)
        assert "loss" in outputs
        assert isinstance(outputs["loss"], torch.Tensor)
        assert outputs["loss"].requires_grad
        assert torch.isfinite(outputs["loss"])
        loss = outputs["loss"]
        print(f"Forward pass successful. Loss: {loss.item()}")
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Optimizer test requires CUDA (can adapt for CPU)")
def test_optimizer_step(initialized_model):
    """Test if loss decreases after one optimizer step."""
    model = initialized_model
    tokenizer = model.tokenizer
    device = torch.device("cuda")
    model.to(device)
    model.train()

    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Create dummy batch (same as test_forward_pass)
    dummy_text = ["<NER> Sample one.", "<NER> Sample two with entity."]
    dummy_target = ["<TAG> 7-10", "<TAG> 20-26"]
    dummy_bio = [[0, 0, 1, 0, -100],[0, 0, 0, 0, 1, -100]]
    inputs = tokenizer(dummy_text, return_tensors="pt", padding="longest", truncation=True, max_length=16)
    targets = tokenizer(dummy_target, return_tensors="pt", padding="longest", truncation=True, max_length=8)
    labels = targets.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    max_len = inputs.input_ids.shape[1]
    token_labels_list = []
    for bio_seq in dummy_bio:
        padded_bio = bio_seq[:max_len] + [-100] * (max_len - len(bio_seq))
        token_labels_list.append(padded_bio)
    token_labels = torch.tensor(token_labels_list, dtype=torch.long)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels.to(device)
    token_labels = token_labels.to(device)

    # First forward pass
    outputs1 = model(**inputs, labels=labels, token_labels=token_labels)
    loss1 = outputs1["loss"]
    assert torch.isfinite(loss1)

    # Optimizer step
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    # Second forward pass
    outputs2 = model(**inputs, labels=labels, token_labels=token_labels)
    loss2 = outputs2["loss"]
    assert torch.isfinite(loss2)

    print(f"Loss before step: {loss1.item():.4f}, Loss after step: {loss2.item():.4f}")
    # Check if loss decreased (allow for slight increase due to numerical instability)
    assert loss2 < loss1 + 1e-3 # Loss should generally decrease 