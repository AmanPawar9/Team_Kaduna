"""
Load Gemma model and tokenizer.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_gemma(model_name="google/gemma-2b", device="cuda"):
    """
    Load the pretrained Gemma model and tokenizer.
    The model is loaded in half-precision to save memory and is set to eval mode.
    Args:
        model_name (str): Hugging Face model identifier.
        device (str): "cuda" or "cpu".
    Returns:
        model, tokenizer
    """
    print(f"Loading Gemma model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Gemma uses a specific tokenizer; ensure padding token is set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,   # half precision to reduce memory
        device_map="auto",           # automatically uses GPU if available
        output_hidden_states=True,   # we need hidden states!
        low_cpu_mem_usage=True
    )
    model.eval()   # freeze – no training
    print("Model loaded successfully.")
    return model, tokenizer
