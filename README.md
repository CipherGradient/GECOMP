# GECOMP
A Black-box textual adversarial attack framework for our paper 《Generative Textual Adversarial Attack through Extensible Compositional Perturbation via Reinforcement Learning for Policy Optimization》

## Install

```bash
cd gecomp
python -m pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger_eng')"
```

Run from the `gecomp/` folder.

## 1. Get models

Generator (rewriter): download [google/flan-t5-base](https://huggingface.co/google/flan-t5-base). Hugging Face will pull it on first run, or put a local copy under `model/`.

Similarity check uses [paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2). Fluency (PPL) uses [gpt2](https://huggingface.co/openai-community/gpt2). Both also auto-download unless you mirror them under `model/`.

More details can be found in our paper.

## 2. Get data

Datasets come from Hugging Face `datasets` and are cached after the first load:

- `sst2` → glue/sst2
- `mr` → rotten_tomatoes
- `ag_news` → ag_news
- `amazon_polarity` → amazon_polarity


## 3. Train

```bash
python -m main.train --dataset sst2 --victim roberta_sst2
```

Checkpoints go to `model/<victim>_<dataset>/` (e.g. `epoch_3`).

## 4. Attack

```bash
python -m main.attack --dataset <dataset> --subset 600 --victim <victim> --planner_ckpt model/<victim>_<dataset>/` (e.g. `epoch_3`)
```

Results are written under `outputs/`.

## Operators

We provide several basic perturbation operators: `misspell`, `homoglyph`, `semantic`, `phonetic`.

## If you want to add a new operator:

Copy `main/ops/_template_op.py` to something like `main/ops/my_op.py`.

Register with `@register("my_op")`. Function shape: `(text, target="") -> (new_text, ok)`.

Import the module in `main/ops/__init__.py`, then pass it in the pool:

```bash
--ops_pool "misspell,homoglyph,semantic,phonetic,my_op"
```

More detail: `main/ops/ADDING_AN_OPERATOR.md`.

## Note

Research use only. Do not attack systems without permission.
