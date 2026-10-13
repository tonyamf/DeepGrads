# DeepGrads

Notebooks written while following Andrej Karpathy's "Neural Networks: Zero to Hero" video lessons (Dec 2022 – Mar 2023).
This is study material that re-implements the lessons, not an original research project.
The data file `names.txt` is the names dataset used in the lessons.

| Notebook | Topic |
|---|---|
| `micrograd.ipynb` | A tiny scalar autograd engine (`Value`) and a small MLP trained on 4 points |
| `BigramMakemore.ipynb` | Character-level bigram model by counting |
| `1LayerNNMakemore.ipynb` | The same bigram model as a one-layer network trained by gradient descent |
| `MLPMakemore.ipynb` | Multi-layer perceptron with embeddings (unfinished; see below) |

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
jupyter lab            # run each notebook top to bottom, from this folder (names.txt is read by relative path)
```

The `Digraph` drawing in `micrograd.ipynb` also needs the Graphviz system program (`dot`), e.g. `apt install graphviz`.

## Status

- `micrograd.ipynb`: runs up to the drawing cells (needs `dot` and `torch`). It also contains scratch cells at the end (manual backward steps) that were left from experimenting; they are not a clean script.
- The three `*Makemore` notebooks need PyTorch. They were not re-run after the fixes because the PyTorch download was too large for the automated run, so their saved outputs are the only results available.
- `MLPMakemore.ipynb` is marked incomplete in the commit message. Its cells are out of order: the 200000-step training loop uses `Xtr`/`Ytr`, which are only created in a later cell, and it indexes `X[ix]` instead of `Xtr[ix]`. Run the `build_dataset` cell before the loop. Not changed here.
- Saved loss numbers in the notebooks (for example bigram average negative log-likelihood 3.6015 on one test word) come from the notebooks' stored outputs, not from a fresh run.
