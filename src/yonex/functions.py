import numpy as np
from yonex import Variable, Function

class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.sin(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        gx = gy * cos(x)
        return gx
def sin(x: Variable) -> Variable:
    return Sin()(x)

class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.cos(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x, = self.inputs
        gx = gy * -sin(x)
        return gx
def cos(x: Variable) -> Variable:
    return Cos()(x)

class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx
def tanh(x: Variable) -> Variable:
    return Tanh()(x)
