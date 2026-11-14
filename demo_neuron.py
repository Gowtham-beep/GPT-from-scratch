from engine import Value
from visualize import draw_dot

# Inputs and weights with labels so they show up clearly in the diagram
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')

x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

# Run backpropagation
o.backward()

# Render and save to neuron_graph.svg (opens cleanly in any browser)
dot = draw_dot(o, format='svg')
output_path = dot.render('neuron_graph', cleanup=True)

print("=== Single Neuron Forward & Backward Pass ===")
print(f"Output (o):       {o.data:.6f} | grad: {o.grad:.4f}")
print(f"n:               {n.data:.6f} | grad: {n.grad:.4f}")
print(f"Weights:         w1={w1.data:.4f} (grad={w1.grad:.4f}), w2={w2.data:.4f} (grad={w2.grad:.4f})")
print(f"Bias:            b={b.data:.4f} (grad={b.grad:.4f})")
print(f"Inputs:          x1={x1.data:.4f} (grad={x1.grad:.4f}), x2={x2.data:.4f} (grad={x2.grad:.4f})")
print(f"\nRendered computation graph saved to: {output_path}")
