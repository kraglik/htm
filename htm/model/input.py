import numpy as np


class InputLayer:
    def __init__(self, shape):
        self.shape = shape
        self.value = np.zeros(shape, dtype=np.bool)

    def set_value(self, value):
        self.value = value
