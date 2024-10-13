import graphviz
from yonex.core import Variable, Function

def _dot_var(dot: graphviz.Digraph, var: Variable, verbose=False) -> None:
    name = str(id(var))
    label = '' if var.name is None else var.name
    if verbose and var.data is not None:
        if var.name is not None:
            label += ': '
        label += str(var.shape) + ' ' + str(var.dtype)
    dot.node(name, label, fillcolor='orange', style='filled')

def _dot_func(dot: graphviz.Digraph, func: Function) -> None:
    # func node
    name = str(id(func))
    label = func.__class__.__name__
    dot.node(name, label, fillcolor='lightblue', style='filled', shape='box')
    # variable node
    for x in func.inputs:
        dot.edge(str(id(x)), name)
    for y in func.outputs:
        dot.edge(name, str(id(y())))

def plot_dot_graph(output: Variable, verbose=True) -> graphviz.Digraph:
    dot = graphviz.Digraph()
    funcs = []
    seen_set = set()
    
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    
    add_func(output.creator)
    _dot_var(dot, output, verbose)
    while funcs:
        func = funcs.pop()
        _dot_func(dot, func)
        for x in func.inputs:
            _dot_var(dot, x, verbose)
            if x.creator is not None:
                add_func(x.creator)
    return dot