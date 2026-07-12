# Team Member 2 — Dataset & Preprocessing

---

# Your Role

Your job is to prepare the data before it is passed to the Gemma model.

Think of your module as the **input pipeline** of the project.

If the data is not clean and consistent, the rest of the project will produce unreliable results.

Your goal is to create a standardized dataset that can be directly used by Team Member 3 for extracting hidden representations.

---

# Where Your Module Fits

```
Raw Dataset
      │
      ▼
Dataset Loading
      │
      ▼
Cleaning & Validation
      │
      ▼
Label Encoding
      │
      ▼
Tokenization
      │
      ▼
Processed Dataset
      │
      ▼
Team Member 3
```

---

# Main Responsibilities

- Load all datasets.
- Clean the text.
- Validate dataset format.
- Standardize labels.
- Split data into Train / Validation / Test.
- Tokenize text using Gemma tokenizer.

---

# Recommended Datasets

Choose **one or two** of the following (depending on course requirements):

| Dataset | Purpose | Link |
|----------|---------|------|
| Dreaddit | Stress Detection | https://huggingface.co/datasets/dreaddit |
| GoEmotions | Emotion Classification | https://huggingface.co/datasets/google-research-datasets/go_emotions |
| Emotion Dataset | Emotion Classification | https://huggingface.co/datasets/dair-ai/emotion |

> If the instructor has specified a dataset, use that instead.

---

# Folder You Own

```
src/data/

src/preprocessing/

src/tokenization/

data/raw/

data/processed/
```

---

# Files to Create

```
loader.py

validator.py

cleaner.py

splitter.py

label_encoder.py

tokenizer.py
```

---

# Expected Functions

### loader.py

```python
load_dataset(path)
```

Loads the dataset into a Pandas DataFrame.

---

### validator.py

```python
validate_dataframe(df)
```

Checks:

- Missing text
- Missing labels
- Duplicate IDs
- Empty rows

---

### cleaner.py

```python
clean_text(text)
```

Suggested cleaning:

- Remove extra spaces
- Remove HTML tags
- Normalize Unicode
- Keep punctuation (important for sentiment)

---

### splitter.py

```python
create_train_val_test_split(df)
```

Recommended split:

- Train: 70%
- Validation: 15%
- Test: 15%

Use **stratified sampling**.

---

### label_encoder.py

```python
encode_labels(df)
```

Convert text labels into integer labels.

Example:

```
Positive → 0
Neutral → 1
Negative → 2
```

---

### tokenizer.py

Use the Gemma tokenizer from Hugging Face.

Function:

```python
tokenize_batch(texts)
```

Output:

- input_ids
- attention_mask

---

# Expected Dataset Format

Every processed dataset should look like:

| id | text | label |
|----|------|-------|
| 1 | I feel hopeless today | Depression |
| 2 | Life is going well | Positive |

---

# Output for Team Member 3

After preprocessing, provide:

```
processed_dataset.csv
```

or

```
processed_dataset.parquet
```

along with tokenized outputs if required.

---

# Testing Checklist

- [ ] Dataset loads correctly
- [ ] No missing values
- [ ] Labels encoded correctly
- [ ] Train/Validation/Test split created
- [ ] Tokenization works

---

# Useful Resources

- Hugging Face Datasets: https://huggingface.co/docs/datasets
- Pandas Documentation: https://pandas.pydata.org/docs/
- Gemma Tokenizer (Transformers): https://huggingface.co/docs/transformers
