from graphviz import Digraph

def trace(root):
    # Builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        uid = str(id(n))
        # Rectangular record node showing label, data, and grad
        label_str = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad)
        dot.node(name=uid, label=label_str, shape='record')
        if n._op:
            # Oval node for the operation (+, *, tanh, etc.)
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # Connect parent Value node to the operation node of child Value
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot