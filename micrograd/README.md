# Micrograd Suite

A tiny scalar-valued autograd engine and neural network library with computational graph visualization, built from scratch.

## Project Structure

```
micrograd/
├── engine.py           # Core autograd engine (Value class with backward pass)
├── nn.py               # Neural network components (Module, Neuron, Layer, MLP)
├── visualize.py        # Graphviz computational graph visualization (draw_dot)
├── demo_neuron.py      # Demo: single neuron graph and backpropagation
├── demo_mlp.py         # Demo: 2-layer MLP single-sample training and SVG snapshots
├── train.py            # Training an MLP on a toy dataset (binary classification)
├── notes.txt           # Implementation notes and reference code
├── neuron_graph.svg    # Rendered computational graph of demo_neuron
├── train_step_0.svg    # Rendered computation graph at step 0 of demo_mlp
└── train_step_19.svg   # Rendered computation graph at step 19 of demo_mlp
```

## Running the Demos

You can run scripts either from the repository root or from inside the `micrograd/` directory:

```bash
# 1. Single neuron demo and graph render
python micrograd/demo_neuron.py

# 2. MLP training demo with visual snapshots
python micrograd/demo_mlp.py

# 3. Full toy dataset training
python micrograd/train.py
```

Or as a Python package from external scripts:

```python
from micrograd import Value, MLP, draw_dot
```
