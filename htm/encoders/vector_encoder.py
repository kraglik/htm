import numpy as np


class VectorEncoder:
    def __init__(
            self,
            size: int,
            minimum: float,
            maximum: float,
            w: int = None,
            n: int = None,
            precision: float = None
    ):
        self.size = size
        self.w = w
        self.n = n
        self.precision = precision
        self.minimum = minimum
        self.maximum = maximum
        self.range = maximum - minimum

        if self.precision and not (self.w and self.n):
            if self.w:
                self.n = self.w * 2 + int(self.range / self.precision)
            else:
                self.w = (self.n - int(self.range / self.precision)) // 2
        else:
            self.precision = self.range / (n - w)

    def __call__(self, values: list):
        arrays = [self._encode_value(value) for value in values]
        return np.concatenate(arrays)

    def _encode_value(self, value: float):
        value = (min(self.maximum, value) - self.minimum) / self.range
        center = int(value * (self.n - self.w)) + self.w // 2
        result = np.zeros(self.n, dtype=np.bool)
        result[center - self.w // 2: center + self.w // 2] = True

        return result
