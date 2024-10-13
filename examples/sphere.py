import numpy as np
from yonex import Variable

def sphere(x: Variable, y: Variable) -> Variable:
    z = x ** 2 + y ** 2
    return z

def main() -> None:
    x = Variable(np.array(1.0), name='x')
    y = Variable(np.array(1.0), name='y')
    z = sphere(x, y)
    z.backward()
    print(x.grad, y.grad)
    
if __name__ == '__main__':
    main()
