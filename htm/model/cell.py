import enum
import uuid

from .cell_state import CellState


class Cell:
    def __init__(self, column):
        self.column = column
        self.id = uuid.uuid4().int
        self.distal_segments = []
        self.apical_segments = []
        self.active = False
        self.was_active = False
        self.predictive = False
        self.was_predictive = False
        self.learning = False
        self.was_learning = False

    @property
    def predicted(self):
        return self.was_predictive and self.active

    def step(self):
        self.was_predictive = self.predictive
        self.was_active = self.active
        self.was_learning = self.learning
        self.learning = False
        self.active = False
        self.predictive = False

        for segment in self.distal_segments:
            segment.step()

    def best_segment(self, config):
        if not self.distal_segments:
            return None

        best_activation, best_segment = 0, None

        for segment in self.distal_segments:
            segment_activation = 0

            for synapse in segment.synapses:
                if synapse.permanence >= config.permanence_threshold and synapse.active:
                    segment_activation += 1

            if segment_activation > best_activation:
                best_activation = segment_activation
                best_segment = segment

        return best_segment

    def __hash__(self):
        return hash(self.id)
