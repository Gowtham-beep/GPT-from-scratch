from engine import Value
from nn import MLP
from visualize import draw_dot

model = MLP(2, [2, 1])
x = [Value(2.0, label='x1'), Value(-1.0, label='x2')]
target = 1.0

print("=== MLP Training (20 Steps) ===")
for step in range(20):
    pred = model(x)
    loss = (pred - target) ** 2
    loss.label = 'loss'

    model.zero_grad()
    loss.backward()

    # Save visual snapshots before updating weights
    if step in (0, 19):
        dot = draw_dot(loss, format='svg')
        saved_file = dot.render(f'train_step_{step}', cleanup=True)
        print(f"[Snapshot] Saved step {step} computation graph to: {saved_file}")

    for p in model.parameters():
        p.data += -0.05 * p.grad

    print(f"Step {step:2d} | Loss: {loss.data:.6f} | Prediction: {pred.data:+.4f} | Target: {target:+.1f}")

print("\nFinished 20 training steps successfully!")
