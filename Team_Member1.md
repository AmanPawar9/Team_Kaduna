# Team Member 1 — Infrastructure, Repository Setup & Common Utilities

---

# Your Role

Welcome!

Your responsibility is **not to build the Machine Learning model**.

Instead, you are responsible for building the **foundation on which the entire project will run.**

Think of yourself as the software architect of the team.

If your module is built correctly,

- everyone else's code will work together easily,
- experiments will be reproducible,
- adding new features will be simple.

Without your work, everyone will be writing code differently and integrating everything at the end will become extremely difficult.

---

# The Big Picture

Our project pipeline looks like this

```

Mental Health Dataset

↓

Dataset Cleaning

↓

Gemma Tokenizer

↓

Gemma Model

↓

Hidden States

↓

Pooling

↓

Sentence Embeddings

↓

Linear Probe

↓

Prediction

↓

Evaluation

```

Notice something.

You are **not responsible** for any of these blocks.

Instead,

you are responsible for everything surrounding them.

```

Repository

↓

Configuration

↓

Logging

↓

Utilities

↓

Environment

↓

Testing

↓

Documentation

↓

GitHub

```

Everything else depends upon your work.

---

# What You Need To Learn

You **DO NOT** need to understand transformers deeply.

Instead, you should become comfortable with

- Python packaging
- GitHub
- Virtual environments
- YAML configuration
- Logging
- Basic testing

---

# Recommended Learning Resources

## Git

https://www.youtube.com/watch?v=RGOj5yH7evk

(FreeCodeCamp Git Tutorial)

---

## Python Packaging

https://packaging.python.org/

---

## Pytest

https://docs.pytest.org/

---

## Logging

https://docs.python.org/3/library/logging.html

---

## Hydra

https://hydra.cc/

---

# Repository Structure

You should create the following repository.

```

latent-probing-gemma/

│

├── configs/

├── data/

│

├── docs/

├── notebooks/

├── outputs/

├── reports/

├── scripts/

├── src/

│

├── tests/

│

├── README.md

├── LICENSE

├── requirements.txt

├── environment.yml

├── pyproject.toml

└── .gitignore

```

---

# Folder Responsibilities

## configs/

Stores every configurable parameter.

Examples

```

dataset.yaml

model.yaml

probe.yaml

evaluation.yaml

```

No hardcoded values should appear in code.

---

## docs/

Contains

- installation guide

- project documentation

- diagrams

- contribution guide

---

## tests/

Contains unit tests.

Every module should eventually have tests.

---

## src/

Contains actual Python code.

No notebooks.

No experiments.

Only reusable code.

---

# Files You Must Create

```

README.md

LICENSE

.gitignore

requirements.txt

environment.yml

pyproject.toml

CONTRIBUTING.md

```

---

# Python Environment

Create

requirements.txt

Include

```

torch

transformers

numpy

pandas

scikit-learn

matplotlib

pytest

hydra-core

tqdm

```

---

Create

environment.yml

using Conda.

---

# Git Ignore

Create a proper

.gitignore

Ignore

```

__pycache__/

.ipynb_checkpoints/

*.pyc

*.pth

*.pt

*.npy

outputs/

logs/

wandb/

mlruns/

```

---

# Configuration System

Every experiment should use YAML.

Example

```

configs/model.yaml

```

```

model:

name: gemma

layer: 12

pooling: mean

```

---

Another example

```

configs/probe.yaml

```

```

probe:

type: logistic

C: 1.0

max_iter: 1000

```

---

Create a configuration loader.

File

```

src/utils/config.py

```

Functions

```

load_config()

```

---

# Logging

Create

```

src/utils/logger.py

```

Functions

```

get_logger()

```

Every module should use

```

logger.info()

logger.warning()

logger.error()

```

instead of print().

---

# Device Utility

Create

```

device.py

```

Function

```

get_device()

```

Should automatically detect

- CUDA

- CPU

- Apple MPS

---

# Random Seed Utility

Create

```

random.py

```

Function

```

set_seed(seed)

```

Should initialize

- random

- numpy

- torch

---

# Path Utility

Avoid hardcoding paths.

Create

```

paths.py

```

Example

```

ROOT

DATA_DIR

MODEL_DIR

REPORT_DIR

OUTPUT_DIR

```

---

# Common Constants

Create

```

constants.py

```

Examples

```

DEFAULT_BATCH_SIZE

DEFAULT_SEED

SUPPORTED_POOLING

SUPPORTED_PROBES

```

---

# Helper Functions

Create

```

helpers.py

```

Examples

```

save_json()

load_json()

create_directory()

save_yaml()

```

---

# Documentation

Create

```

docs/

Installation.md

ProjectStructure.md

```

Explain

- how to install

- folder layout

- repository philosophy

---

# GitHub Workflow

Every teammate should create

their own branch.

Examples

```

member1

member2

member3

member4

member5

```

Nobody pushes directly

to

main.

---

# Pull Requests

Before merging,

check

- Code runs

- Documentation updated

- No merge conflicts

---

# Unit Tests

Create

```

tests/

```

Write

basic tests

for

```

config loader

logger

paths

seed

```

---

# Coding Style

Please ensure

- meaningful variable names

- functions are short

- no repeated code

- comments where needed

---

# What Should NOT Be Done

Do NOT

- train models

- preprocess data

- extract embeddings

- implement probes

- calculate metrics

Those belong to other members.

---

# Interaction With Other Members

## Member 2

Needs

- configuration

- paths

- logging

---

## Member 3

Needs

- utilities

- configuration

- device manager

---

## Member 4

Needs

- configuration

- logging

---

## Member 5

Needs

- output directories

- logging

---

# Deliverables Checklist

Repository Structure

- [ ] Repository created
- [ ] Folder structure completed
- [ ] GitHub initialized

Environment

- [ ] requirements.txt
- [ ] environment.yml

Utilities

- [ ] logger.py
- [ ] config.py
- [ ] paths.py
- [ ] helpers.py
- [ ] constants.py
- [ ] random.py
- [ ] device.py

Documentation

- [ ] Installation guide
- [ ] Project structure guide
- [ ] Contribution guide

Testing

- [ ] Utility tests
- [ ] Configuration tests

---

# Definition of Done

Your work is complete when

✅ Everyone can clone the repository.

✅ Everyone can install the environment.

✅ Everyone can import the project.

✅ Configuration files work.

✅ Logger works.

✅ Utility functions work.

✅ Documentation is complete.

At this point,

the remaining four members should be able to start developing independently without modifying your code.

---

# Helpful References

## Git

https://git-scm.com/docs

---

## Hydra

https://hydra.cc/docs/intro/

---

## PyTorch Installation

https://pytorch.org/get-started/

---

## HuggingFace

https://huggingface.co/docs

---

## Pytest

https://docs.pytest.org/

---

# Final Goal

Although your contribution does not involve Machine Learning directly, it is one of the most important parts of the project.

A well-organized repository makes collaboration easier, reduces bugs, improves reproducibility, and allows the rest of the team to focus entirely on the research components of the project.
