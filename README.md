# Transformer: PyTorch Implementation of "Attention Is All You Need"

This repository provides a comprehensive PyTorch implementation of the Transformer model introduced in the seminal paper [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Vaswani et al., NIPS 2017).

<p align="center">
  <img src="https://github.com/user-attachments/assets/c03ff638-94b2-47c4-8061-ec38e371b99d" alt="Transformer Model Architecture" width="471"/>
</p>

## Overview

The Transformer represents a paradigm shift in sequence-to-sequence modeling, utilizing attention mechanisms exclusively without relying on recurrent (RNN) or convolutional (CNN) architectures. This innovative design offers several key advantages:

- **Enhanced Performance**: Achieves state-of-the-art results on translation benchmarks
- **Parallelization**: Enables significantly faster training compared to sequential models
- **Long-Range Dependencies**: More effectively captures relationships between distant elements in sequences

## Architecture

The Transformer architecture consists of an encoder-decoder structure with several distinctive components:

### Key Components

- **Multi-Head Attention**: Allows the model to jointly attend to information from different representation subspaces
- **Self-Attention**: Computes relationships between all positions in a sequence 
- **Positional Encoding**: Incorporates sequential information through sinusoidal position embeddings
- **Layer Normalization & Residual Connections**: Facilitates training of deep networks

### Core Formula

The scaled dot-product attention mechanism is defined as:

```
Attention(Q, K, V) = softmax(QK^T/√d_k)V
```

Where Q, K, and V represent the queries, keys, and values matrices respectively, and d_k is the dimension of the keys.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f278d23-bb14-47e1-851e-b4b1d1198857" alt="Multi-Head Attention" width="651"/>
</p>

## Performance

The original Transformer architecture achieved breakthrough results:

| Dataset | Metric | Score |
|---------|--------|-------|
| WMT 2014 English-German | BLEU | 28.4 |
| WMT 2014 English-French | BLEU | 41.0 |

These results were achieved with significantly reduced training time compared to previous state-of-the-art models.

## Implementation Features

This PyTorch implementation includes:

- Clean, modular, and well-documented code
- Full encoder-decoder architecture with self-attention and feed-forward networks
- Positional encoding and multi-head attention mechanisms
- Configurable hyperparameters matching the original paper specifications
- Training and inference pipelines with example usage

## Requirements

```
pytorch >= 1.7.0
numpy >= 1.19.0
tqdm
matplotlib (for visualizations)
```

## Usage

```python
from transformer import Transformer

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=100,
    dropout=0.1
)

# Training and inference examples in examples/
```

## References

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Vaswani et al., 2017)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) (Harvard NLP)
- [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) (Original TensorFlow Implementation)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
