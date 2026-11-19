# GPT from Scratch

Building generative pre-trained transformers and modern neural network architectures from first principles.

## Roadmap & Modules

1. **[micrograd/](file:///home/gpuserver1/projects/GPT/micrograd)**: Scalar autograd engine, backpropagation, MLP neural networks, and automated computation graph visualization.
   - See the full [Micrograd Documentation & Visual Guide](file:///home/gpuserver1/projects/GPT/micrograd/README.md) for detailed mathematical derivations and computational graph diagrams.
2. **Upcoming Modules**:
   - `makemore`: Autoregressive character-level language modeling (Bigram, MLP, BatchNorm, WaveNet).
   - `nanoGPT`: Transformer attention mechanisms, positional encodings, multi-head self-attention, and GPT language model training.

## Quickstart

```bash
# Activate environment
source .venv/bin/activate

# Run the micrograd suite
python micrograd/demo_neuron.py
python micrograd/demo_mlp.py
python micrograd/train.py
```
