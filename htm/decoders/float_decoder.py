import sklearn.neural_network as nn
import numpy as np


class FloatDecoder:
    def __init__(self, region, steps_ahead: int):
        self.region = region
        self.steps_ahead = steps_ahead
        self.input_size = region.sdr.flatten().shape[0]
        self._nn = nn.MLPRegressor(
            solver='sgd',
            hidden_layer_sizes=50,
            max_iter=10,
            nesterovs_momentum=True,
            activation='logistic'
        ).fit([self.region.sdr.flatten()], [0.0])
        self._history = []

    def __call__(self, value) -> float:
        sdr = self.region.sdr.reshape((self.input_size,)).astype(np.float)
        self._history.append(sdr)

        try:
            prediction = self._nn.predict(sdr.reshape(1, -1))[0]
        except Exception as e:
            prediction = 0.0

        if len(self._history) > self.steps_ahead:
            input = self._history.pop(0)
            self._nn = self._nn.partial_fit([input], [value])
            self._nn = self._nn.partial_fit([input], [value])
            self._nn = self._nn.partial_fit([input], [value])

        return prediction


