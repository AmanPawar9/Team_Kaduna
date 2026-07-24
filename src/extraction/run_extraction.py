"""
Run extraction on a dataset and save embeddings.
"""
import os
import numpy as np
import torch
from models.gemma_loader import load_gemma
from pooling.strategies import mean_pooling, max_pooling, cls_pooling
from extraction.hidden_extractor import extract_hidden_representations
from utils.config_loader import load_config
from utils.helpers import ensure_dir, set_seed

def main():
    # Load configuration
    config = load_config("configs/config.yaml")
    set_seed(config["random_seed"])

    # ---- Load a dummy dataset for testing ----
    # In the real project, Member 2 will provide the cleaned dataset.
    # For now, we create some mental health related texts.
    sample_texts = [
        "I feel extremely anxious and worried all the time.",
        "I am so happy and grateful for everything.",
        "I have been feeling depressed and hopeless lately.",
        "Life is wonderful and I love every moment.",
        "I am struggling with panic attacks every day.",
        "Everything is going great, I feel fantastic!"
    ]
    # You can also load a real dataset from Hugging Face, e.g.,:
    # from datasets import load_dataset
    # dataset = load_dataset("your_mental_health_dataset")
    # texts = dataset["train"]["text"]

    # ---- Load model and tokenizer ----
    model, tokenizer = load_gemma(config["model_name"])

    # ---- Pooling strategies to try ----
    pooling_strategies = {
        "mean": mean_pooling,
        "max": max_pooling,
        "cls": cls_pooling
    }

    # ---- Extract embeddings for each pooling strategy ----
    layers = config["layers_to_extract"]
    for strategy_name, pooling_fn in pooling_strategies.items():
        print(f"\nExtracting with {strategy_name} pooling ...")
        embeddings = extract_hidden_representations(
            model=model,
            tokenizer=tokenizer,
            texts=sample_texts,
            layers=layers,
            pooling_fn=pooling_fn,
            batch_size=config["batch_size"],
            max_length=config["max_length"]
        )
        # Save embeddings
        save_path = os.path.join(config["embeddings_path"], f"embeddings_{strategy_name}.npz")
        ensure_dir(os.path.dirname(save_path))
        np.savez_compressed(save_path, **embeddings)
        print(f"Saved embeddings to {save_path}")
        print(f"Shape for layer {layers[0]}: {embeddings[layers[0]].shape}")

    print("\n✅ Extraction complete!")

if __name__ == "__main__":
    main()
