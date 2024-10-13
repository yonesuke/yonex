import numpy as np
from yonex import Variable
import yonex.functions as F
from yonex.utils import plot_dot_graph

def main() -> None:
    x = Variable(np.array(1.0), name='x')
    y = F.tanh(x)
    y.name = 'y'
    y.backward(create_graph=True)
    
    iters = 3
    for _ in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
    
    gx = x.grad
    gx.name = 'gx' + str(iters + 1)
    dot = plot_dot_graph(gx, verbose=False)
    dot.render('tanh', directory='figure', format='png', cleanup=True)

if __name__ == '__main__':
    main()
