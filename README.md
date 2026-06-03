# Transformers From Scratch
## Overview

Implements a transformer encoder-decoder architecture from first principles using NumPy. The implementation is modular: core mathematical operations are separated into small, focused files so the attention math, normalization, and feed-forward logic can be examined independently.

## Structure

- model/: building blocks and assembly of the transformer (attention, FFN, layer norm, encoder/decoder blocks, and the transformer wrapper).
- data/: a conceptual toy pipeline for tokenization, vocabulary lookup, pairing, and batching. Designed for demonstration rather than production use.
- utils/: small helper modules (positional encodings and activation functions).

## Model components

- Multi-head attention: query/key/value projections, head splitting, scaled dot-product attention, optional masking, and output projection.
- Feed-forward network (FFN): a two-layer position-wise MLP with ReLU activation.
- Layer normalization: normalization along the feature dimension to stabilize residual connections.
- Encoder block: self-attention + residual + layer norm + feed-forward + residual + layer norm.
- Decoder block: masked self-attention, cross-attention over encoder outputs, feed-forward, and normalization steps.
- Transformer wrapper: stacks encoder and decoder blocks to produce a full encoder-decoder model.

## Data pipeline (conceptual)

The data modules present a minimal pipeline that demonstrates shape and batching requirements for training a transformer:

- tokenization by whitespace
- fixed source and target vocabularies (small, hard-coded)
- converting sentence pairs into source ids, target input ids (with a start token), and target output ids (with an end token)
- a simple batch iterator that pads sequences to batch width and returns NumPy arrays

This design is intended as a teaching scaffold. For production or training at scale, the conceptual components in data/ would typically be replaced by a PyTorch `Dataset`, `DataLoader`, and collate function producing tensors.


## Quick smoke tests

Several model files include small `__main__` examples that run forward passes. From the repository root, the examples can be run directly, for example:

```bash
python model/encoder.py
python model/decoder.py
python model/transformer.py
```

The data pipeline can be inspected by importing or running the data utilities (for demonstration only):

```bash
python data/dataset.py
python data/loader.py
```

## Migration notes (PyTorch direction)

To move this implementation into a typical PyTorch workflow, the following steps are common:

- replace NumPy operations with `torch` tensors and functions
- convert attention, FFN, normalization, encoder, and decoder into `torch.nn.Module` classes
- use real embedding layers and `torch.utils.data` utilities for data loading
- add a training script with loss, optimizer, and checkpointing

## How to explore the code

Recommended reading order for understanding the implementation:

1. data/vocab.py — see how text is tokenized and converted to ids
2. model/Attention.py — inspect query/key/value projection and attention math
3. model/FFN.py and model/LayerNorm.py — supporting components used in blocks
4. model/encoder.py and model/decoder.py — assembly of layer-level logic
5. model/transformer.py — stacking encoder and decoder into the full model

## Summary

 This repositoty provides a concise, modular, NumPy-based transformer implementation intended for learning and experimentation. The data folder is a conceptual scaffold that illustrates data shapes and batching; replacing it with a PyTorch-style data pipeline and adding a training loop are the natural next steps for extending this work.

