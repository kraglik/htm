from htm.model.cell_state import CellState


class ApicalSynapse:
    def __init__(self, input_cell, owner_segment):
        self.input_cell = input_cell
        self.owner_segment = owner_segment
        self.permanence = 0.0
        self.active = False


class DistalSynapse:
    def __init__(self, input_cell, owner_segment):
        self.input_cell = input_cell
        self.owner_segment = owner_segment
        self.permanence = 0.0
        self.actual = False

    @property
    def active(self):
        return not self.input_cell.active

    @property
    def was_active(self):
        return self.input_cell.was_active


class ProximalSynapse:
    def __init__(self, input_layer, input_coords, owner_segment, permanence=0.0):
        self.input_layer = input_layer
        self.input_coords = input_coords
        self.owner_segment = owner_segment
        self.permanence = permanence

    @property
    def is_active(self):
        return bool(self.input_layer.value[self.input_coords])
