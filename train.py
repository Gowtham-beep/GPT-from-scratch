from nn import MLP
from engine import Value

# sanity check the engine
a = Value(3.0)
b = a + a
b.backward()
assert a.grad == 2.0, f"Expected grad 2.0, got {a.grad}"

#Define dataset ( 4 examples , 3 features each ) and desired targets
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# Initialize a 3 -> 4 -> 4 -> 1 neural network (41 parameters)
model = MLP( 3,[4,4,1])
print(f"Total parameters: {len(model.parameters())}")

#Gradient descent training loop
learning_rate = 0.05
for step in range(50):
    #Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    #zero gradients
    model.zero_grad()

    #Backward pass
    loss.backward()

    #update parameters
    for p in model.parameters():
        p.data += -learning_rate * p.grad

    if step % 5 == 0 or step == 49:
        print(f"Step {step:2d} | Loss: {loss.data:.6f}")

print("\nFinal predictions vs targets:")
for pred, target in zip(ypred, ys):
    print(f"Pred: {pred.data:+.4f} | Target: {target:+.1f}")



