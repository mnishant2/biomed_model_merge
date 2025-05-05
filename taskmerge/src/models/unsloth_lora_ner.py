# Placeholder for Unsloth LoRA NER Model 

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel
from unsloth import FastLlamaModel # Use FastLlamaModel for Llama architectures
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class UnslothLoRANER(nn.Module):
    """Unsloth-optimized Llama3 model with LoRA for generative NER span prediction.

    Includes an optional token classification head for auxiliary metrics.
    """
    def __init__(self,
                 base_model_id: str = "aaditya/Llama3-OpenBioLLM-8B",
                 tokenizer_path: str = "taskmerge/tokenizers/llama3_biomerge_tok",
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 target_modules: str = "all-linear", # Common practice, adjust if needed based on unsloth docs/exp
                 add_token_head: bool = True,
                 dtype_str: str = "bf16",
                 load_in_4bit: bool = False, # Spec says no 4-bit
                 gradient_checkpointing: bool = True):
        super().__init__()

        self.base_model_id = base_model_id
        self.tokenizer_path = tokenizer_path
        self.add_token_head = add_token_head
        self.dtype = getattr(torch, dtype_str) if dtype_str else None

        logger.info(f"Initializing Unsloth model: {base_model_id}")
        logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, targets='{target_modules}'")
        logger.info(f"Settings: dtype={dtype_str}, 4-bit={load_in_4bit}, grad_ckpt={gradient_checkpointing}, token_head={add_token_head}")

        # Load model and tokenizer using Unsloth's FastLlamaModel
        self.backbone, self.tokenizer = FastLlamaModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=None, # Handled by tokenizer in datamodule
            dtype=self.dtype,
            load_in_4bit=load_in_4bit,
            # token = "hf_...", # Token is optional if model is public
            # Needs pre-saved tokenizer if custom tokens were added
            # We assume the tokenizer is correctly saved at tokenizer_path
            # If issues arise, might need to load tokenizer separately first
            # and pass it during model init if Unsloth doesn't pick up changes.
        )

        logger.info("Applying LoRA configuration...")
        # Apply LoRA using PEFT
        self.backbone = FastLlamaModel.get_peft_model(
            self.backbone,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",  # Recommended practice
            use_gradient_checkpointing=gradient_checkpointing, # Unsloth handles this
            random_state=42,
            target_modules=target_modules,
            max_seq_length=None, # Already handled
        )

        # Note: The spec mentions self.backbone.enable_gradient_checkpointing().
        # Unsloth's get_peft_model with use_gradient_checkpointing=True should handle this.
        # Verify if direct call is still needed or preferred.
        if gradient_checkpointing and hasattr(self.backbone, 'enable_input_require_grads'):
             # May be needed for gradient checkpointing compatibility
             self.backbone.enable_input_require_grads()


        if add_token_head:
            # Get hidden size from the *base model*'s config before PEFT wrapping if possible
            # Or access through peft model's config
            if hasattr(self.backbone, 'config'):
                hidden_size = self.backbone.config.hidden_size
            else:
                # Fallback or load config separately if needed
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(base_model_id)
                hidden_size = config.hidden_size

            logger.info(f"Adding token classification head (hidden_size={hidden_size})")
            self.token_head = nn.Linear(hidden_size, 3) # O/B/I
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            logger.info("Skipping token classification head.")
            self.token_head = None
            self.ce_loss = None

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, # For Causal LM loss
                token_labels: Optional[torch.Tensor] = None # For optional Token Classification loss
                ):
        """Forward pass for training.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            labels: Target token IDs (shifted inside the model) for Causal LM loss.
            token_labels: Target BIO labels for the optional token classification head.

        Returns:
            Dictionary containing total loss and model logits.
        """
        # Pass labels to the backbone model for Causal LM loss calculation
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=self.add_token_head, # Only need hidden states if head exists
            return_dict=True
        )

        causal_lm_loss = outputs.loss
        total_loss = causal_lm_loss
        token_head_loss = None

        # Calculate token classification loss if head exists and labels are provided
        if self.token_head is not None and token_labels is not None:
            last_hidden_state = outputs.hidden_states[-1]
            # Ensure hidden state is float32 for the linear layer if needed
            token_logits = self.token_head(last_hidden_state.to(self.token_head.weight.dtype))

            # Flatten logits and labels for CrossEntropyLoss
            # Logits: (batch_size, seq_len, num_classes) -> (batch_size * seq_len, num_classes)
            # Labels: (batch_size, seq_len) -> (batch_size * seq_len)
            loss_fct = self.ce_loss
            token_head_loss = loss_fct(token_logits.view(-1, 3), token_labels.view(-1))

            # Add token head loss to the total loss
            # Consider weighting this loss if needed
            total_loss = total_loss + token_head_loss
        elif self.token_head is not None and token_labels is None:
             logger.warning("Token head exists but token_labels were not provided to forward pass.")


        output_dict = {
            "loss": total_loss,
            "logits": outputs.logits, # Logits from the base LM head
            "causal_lm_loss": causal_lm_loss
        }
        if token_head_loss is not None:
            output_dict["token_head_loss"] = token_head_loss
        if self.add_token_head:
             # Also return token logits if head exists, might be useful for eval callbacks
             if 'token_logits' in locals():
                 output_dict["token_logits"] = token_logits

        return output_dict

# Example usage / basic test
if __name__ == '__main__':
    logger.basicConfig(level=logging.INFO)
    print("Testing UnslothLoRANER Initialization...")

    # Assume tokenizer is prepared
    tokenizer_path = "../tokenizers/llama3_biomerge_tok" # Adjust relative path
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}, skipping model init test.")
    else:
        try:
            model = UnslothLoRANER(
                tokenizer_path=tokenizer_path,
                add_token_head=True,
                # Use smaller params for quick local test if needed, but stick to spec for actual use
                # lora_r=2,
                # lora_alpha=4,
            )
            print("Model initialized successfully.")
            print("Backbone:", type(model.backbone))
            # print(model.backbone) # Print model structure (can be long)

            if model.token_head:
                print("Token head initialized:", model.token_head)

            # Test forward pass with dummy data
            print("\nTesting forward pass...")
            tokenizer = model.tokenizer
            dummy_input = "<NER> This is a test sentence." # Example input
            dummy_target = "<TAG> 15-19" # Example target
            dummy_bio = [0, 0, 0, 0, 1, 0, 0, 0] # Example BIO labels (O O O O B O O O)

            inputs = tokenizer(dummy_input, return_tensors="pt", padding="max_length", max_length=64)
            targets = tokenizer(dummy_target, return_tensors="pt", padding="max_length", max_length=16)
            labels = targets.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            token_labels = torch.tensor([dummy_bio + [-100]*(64-len(dummy_bio))], dtype=torch.long)

            # Move to GPU if available for a more realistic test
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            token_labels = token_labels.to(device)

            print(f"Input shape: {inputs['input_ids'].shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Token Labels shape: {token_labels.shape}")

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
                token_labels=token_labels
            )

            print("\nForward pass outputs:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor shape {value.shape}, device {value.device}, dtype {value.dtype}")
                    if value.numel() == 1:
                        print(f"    Value: {value.item():.4f}")
                else:
                     print(f"  {key}: {value}")

            assert "loss" in outputs
            assert "logits" in outputs
            assert outputs["loss"].requires_grad # Ensure loss requires grad for backprop
            print("\nForward pass test completed.")

        except ImportError as e:
            print(f"Import error: {e}. Make sure unsloth and necessary libraries are installed.")
        except Exception as e:
            print(f"An error occurred during model testing: {e}")
            import traceback
            traceback.print_exc() 