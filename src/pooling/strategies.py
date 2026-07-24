"""
Pooling strategies to convert token-level hidden states into sentence embeddings.
"""
import torch

def mean_pooling(hidden_states, attention_mask):
    """
    Average the hidden states across the sequence length, masking padding.
    Args:
        hidden_states: (batch_size, seq_len, hidden_dim)
        attention_mask: (batch_size, seq_len) with 1 for real tokens, 0 for padding.
    Returns:
        pooled: (batch_size, hidden_dim)
    """
    # Expand mask to (batch_size, seq_len, 1) for broadcasting
    mask = attention_mask.unsqueeze(-1).float()
    # Sum hidden states weighted by mask
    sum_embeddings = torch.sum(hidden_states * mask, dim=1)
    # Sum of mask (number of real tokens per sequence)
    sum_mask = torch.sum(mask, dim=1)
    # Avoid division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask

def max_pooling(hidden_states, attention_mask):
    """
    Take the maximum over the sequence length, ignoring padding.
    """
    mask = attention_mask.unsqueeze(-1).float()
    # Set padding tokens to a very large negative value so they are ignored in max
    hidden_states = hidden_states + (1 - mask) * (-1e9)
    pooled, _ = torch.max(hidden_states, dim=1)
    return pooled

def cls_pooling(hidden_states):
    """
    Use the first token's hidden state (CLS token or the first token of the sequence).
    For Gemma (causal LM), the first token is usually the BOS token.
    """
    return hidden_states[:, 0, :]
