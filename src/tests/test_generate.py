# Placeholder for generation tests 

import pytest
import torch
import os
import hydra
from omegaconf import OmegaConf

# Reuse fixtures from test_forward if possible
CONF_PATH = "conf"

@pytest.fixture(scope="module")
def cfg_gen(): # Renamed to avoid conflict if running all tests
    """Load Hydra config for generation testing."""
    hydra.initialize(config_path=os.path.relpath(CONF_PATH, "."), version_base=None)
    cfg = hydra.compose(config_name="config", overrides=[
        "model=unsloth_lora_bf16",
        "task=ner_span",
        "runtime=a100",
        f"model.tokenizer_path=tokenizers/llama3_biomerge_tok",
        "model.lora_r=2",
        "model.lora_alpha=4",
        "++runtime.training.bf16=false", # Test on CPU or non-bf16 GPU
    ])
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    return cfg

@pytest.fixture(scope="module")
def initialized_model_gen(cfg_gen): # Renamed fixture
    """Initialize the model for testing generation."""
    tokenizer_path = cfg_gen.model.tokenizer_path
    if not os.path.exists(tokenizer_path):
        pytest.skip(f"Tokenizer not found at {tokenizer_path}, cannot initialize model.")

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        cfg_gen.model.dtype_str = None
        cfg_gen.model.load_in_4bit = False

    try:
        # Instantiate the nn.Module wrapper
        model_wrapper = hydra.utils.instantiate(cfg_gen.model)
        # Return the underlying backbone model suitable for .generate()
        return model_wrapper.backbone, model_wrapper.tokenizer
    except Exception as e:
        pytest.fail(f"Failed to initialize model: {e}")

# Skip if CUDA not available, as generation is slow on CPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Generation test requires CUDA")
def test_generate_output(initialized_model_gen):
    """Test if model.generate() produces a string starting with <TAG>."""
    model, tokenizer = initialized_model_gen
    device = torch.device("cuda")
    model.to(device)
    model.eval() # Set model to evaluation mode

    # Simple input prompt
    input_text = "<NER> Aspirin is used for pain relief."
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Perform generation
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=16, # Generate a small number of tokens
                do_sample=False, # Greedy
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        # Decode the generated tokens (excluding the input prompt)
        output_text = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Generated text: '{output_text}'")

        assert isinstance(output_text, str)
        # Check if it starts with the expected tag (allow for minor variations/whitespace)
        assert output_text.strip().startswith("<TAG>")

    except Exception as e:
        pytest.fail(f"Generation failed: {e}") 