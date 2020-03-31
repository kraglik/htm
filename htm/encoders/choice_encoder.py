import numpy as np


class ChoiceEncoder:
    def __init__(
            self,
            n_choices: int,
            choice_w: int,
            overlap: int = 0
    ):
        self.n = choice_w * n_choices
        self.options = n_choices
        self.w = choice_w
        self.overlap = overlap

        self.starts = [i * choice_w for i in range(n_choices)]

        if overlap > 0:
            border_w = choice_w - overlap
            medium_w = choice_w - overlap * 2

            self.starts = [0]

            shift = border_w

            for n in range(1, self.options - 1):
                self.starts.append(shift)
                shift += medium_w

            self.starts.append(shift + overlap)

            self.n = shift + overlap + self.w

    def __call__(self, value: int):
        start = self.starts[value - 1]

        result = np.zeros(self.n, dtype=np.bool)
        result[start: start + self.w] = True

        return result
