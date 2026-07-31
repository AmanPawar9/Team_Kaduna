"""
Extract hidden states from Gemma for a given batch of texts.
"""
import torch
from tqdm import tqdm

def extract_hidden_representations(model, tokenizer, texts, layers, pooling_fn, batch_size=8, max_length=512):
    """
    Extract pooled hidden representations from specified layers.
    Args:
        model: Gemma model.
        tokenizer: tokenizer.
        texts: list of strings.
        layers: list of layer indices (e.g., [0, 6, 12, 18]).
        pooling_fn: a function from pooling/strategies.py (e.g., mean_pooling).
        batch_size: int.
        max_length: int.
    Returns:
        embeddings: dict mapping layer_index -> numpy array of shape (num_texts, hidden_dim)
    """
    model.eval()
    device = next(model.parameters()).device
    all_embeddings = {layer: [] for layer in layers}

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
        batch_texts = texts[i:i+batch_size]
        # Tokenize with padding and truncation
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # outputs.hidden_states is a tuple of (num_layers+1) tensors (including embeddings)
            # The first element is the embedding layer, then each transformer layer.
            # We want the layer outputs after the specified layer index.
            # Usually, layer 0 is after the first transformer block, so we may use index+1.
            # But we'll let the user specify the exact tuple index.
            # To be consistent, we assume layers correspond to the transformer layer index,
            # so we take hidden_states[layer+1] (because hidden_states[0] is embedding).
            for layer in layers:
                # hidden_states is tuple; index 0: embeddings, index 1: after layer 0, etc.
                layer_hidden = outputs.hidden_states[layer + 1]  # (batch, seq_len, hidden_dim)
                # Apply pooling
                pooled = pooling_fn(layer_hidden, inputs['attention_mask'])  # (batch, hidden_dim)
                all_embeddings[layer].append(pooled.cpu().numpy())

    # Concatenate all batches
    for layer in layers:
        all_embeddings[layer] = np.concatenate(all_embeddings[layer], axis=0)
    return all_embeddings
