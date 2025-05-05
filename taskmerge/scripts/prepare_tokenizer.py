from transformers import AutoTokenizer
import os

def prepare_tokenizer(base_model_id="aaditya/Llama3-OpenBioLLM-8B", output_dir="taskmerge/tokenizers/llama3_biomerge_tok"):
    """Loads a base tokenizer, adds special tokens, sets padding, and saves.

    Args:
        base_model_id (str): Hugging Face model ID for the base tokenizer.
        output_dir (str): Directory to save the modified tokenizer.
    """
    print(f"Loading base tokenizer: {base_model_id}")
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    special_tokens = ["<NER>", "<TAG>", "<O>", "<B>", "<I>"]
    tokens_to_add = [t for t in special_tokens if t not in tok.vocab]

    if tokens_to_add:
        print(f"Adding special tokens: {tokens_to_add}")
        tok.add_tokens(tokens_to_add)
    else:
        print("Special tokens already present in tokenizer vocabulary.")

    if tok.pad_token is None:
        print("Setting pad_token to eos_token.")
        tok.pad_token = tok.eos_token
    else:
        print(f"Pad token already set: {tok.pad_token}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving tokenizer to: {output_dir}")
    tok.save_pretrained(output_dir)
    print("Tokenizer preparation complete.")

if __name__ == "__main__":
    prepare_tokenizer() 