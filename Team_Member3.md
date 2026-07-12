# Team Member 3 – Gemma Model & Hidden State Extraction

---

# Your Role

This is the **core research module** of the project.

Your job is to use Google's **Gemma** model to generate **latent representations (hidden states)** from text.

These hidden representations will later be used by Team Member 4 to train simple machine learning models (linear probes).

Think of your module as the **feature extractor** of the project.

---

# Where Your Module Fits

```
Processed Dataset
        │
        ▼
Gemma Tokenizer
        │
        ▼
Gemma Model
        │
        ▼
Hidden States
        │
        ▼
Pooling
        │
        ▼
Sentence Embeddings
        │
        ▼
Saved Embeddings
        │
        ▼
Team Member 4
```

---

# Your Objectives

You should be able to:

- Load the pretrained Gemma model.
- Tokenize text correctly.
- Run inference (no training).
- Extract hidden states from every transformer layer.
- Experiment with different transformer layers.
- Convert token embeddings into one sentence embedding.
- Save embeddings for future experiments.

**Important:** We are **not fine-tuning Gemma**. The model remains frozen throughout the project.

---

# Folder Ownership

You are responsible for the following folders:

```
src/models/

src/extraction/

src/pooling/

data/embeddings/
```

---

# Files You Need to Create

```
src/models/
│
├── gemma_loader.py

src/extraction/
│
├── hidden_state_extractor.py
├── layer_selector.py
└── embedding_store.py

src/pooling/
│
├── mean_pool.py
├── last_token_pool.py
├── max_pool.py
└── pool_factory.py
```

---

# Step 1 – Load Gemma

Create:

```
src/models/gemma_loader.py
```

Functions to implement:

```python
load_model()

load_tokenizer()

load_model_and_tokenizer()
```

The model should:

- Load from Hugging Face.
- Automatically detect GPU if available.
- Return both tokenizer and model.

Suggested libraries:

```python
from transformers import AutoTokenizer
from transformers import AutoModel
```

---

# Step 2 – Tokenization

Use the tokenizer provided by Gemma.

Input:

```text
"I don't feel like myself anymore."
```

Output should include:

- input_ids
- attention_mask

Use:

- padding=True
- truncation=True
- max_length=256 (or configurable)

---

# Step 3 – Forward Pass

Run the tokenized inputs through Gemma.

Important:

Enable hidden states.

The model should return:

```
Output

├── logits (ignore)

└── hidden_states
```

---

# Step 4 – Hidden State Extraction

Create

```
hidden_state_extractor.py
```

Function

```python
extract_hidden_states(model, tokenizer, texts)
```

Output:

A list (or tuple) of hidden states.

Each hidden state has the shape:

```
(batch_size,
 sequence_length,
 hidden_dimension)
```

Example:

```
(32,
128,
2304)
```

---

# Step 5 – Layer Selection

Create

```
layer_selector.py
```

Function

```python
select_layer(hidden_states, layer_index)
```

Example:

```
Layer 0

Layer 5

Layer 10

Layer 15

Layer 20

Final Layer
```

We want to compare different layers later.

---

# Step 6 – Pooling

Gemma outputs one embedding per token.

However, our classifiers require one embedding per sentence.

Implement the following pooling methods.

---

## Mean Pooling

Average all token embeddings.

```
Token1

Token2

Token3

↓

Average

↓

Sentence Embedding
```

---

## Last Token Pooling

Use only the last token representation.

---

## Max Pooling (Optional)

Take the maximum value across tokens.

---

# Step 7 – Embedding Storage

Create

```
embedding_store.py
```

Functions:

```python
save_embeddings()

load_embeddings()
```

Recommended formats:

- `.npy`
- `.pt`

Each saved embedding file should include metadata such as:

- Dataset name
- Gemma model version
- Layer number
- Pooling method

---

# Expected Output

For every sentence, produce one embedding.

Example:

Input:

```
"I feel exhausted."
```

Output:

```
[0.213,
-0.551,
0.882,
...
]
```

Dimension depends on the selected Gemma model.

---

# Interface for Team Member 4

Save:

```
embeddings.npy

labels.npy
```

These files will be used directly for training probes.

---

# Testing Checklist

Before handing over your work:

- [ ] Gemma loads successfully.
- [ ] Tokenizer works.
- [ ] Hidden states extracted.
- [ ] Different layers selectable.
- [ ] Mean pooling works.
- [ ] Last-token pooling works.
- [ ] Embeddings saved and reloaded correctly.

---

# Helpful Resources

### Gemma Model

https://huggingface.co/google

### Hugging Face Transformers

https://huggingface.co/docs/transformers

### Model Outputs

https://huggingface.co/docs/transformers/main_classes/output

### PyTorch

https://pytorch.org/docs/stable/

---

# Deliverables

By the end of your module, you should provide:

- Working Gemma loader
- Hidden state extraction code
- Pooling implementations
- Saved embedding files
- Documentation explaining how embeddings are generated

These outputs will be passed directly to Team Member 4 for probe training.
