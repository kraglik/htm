import math
import uuid

from htm.model.cell import Cell
from htm.model.cell_state import CellState


class Column:
    def __init__(self, region, coord, n_cells):
        self.region = region
        self.coord = coord
        self.id = uuid.uuid4().int
        self.cells = [Cell(self, (coord[0], coord[1], i)) for i in range(n_cells)]
        self.cells_ids = [cell.id for cell in self.cells]
        self.active = False
        self.predicted = False
        self.bursting = False
        self.proximal_segments = []
        self.overlap = 0.0
        self.boost = 1.0
        self.activity_duty_cycle = 0.0
        self.overlap_duty_cycle = 0.0

    def step(self):
        self.active = False
        self.predicted = False
        self.bursting = False

        for cell in self.cells:
            cell.step()

    def update_duty_cycles(self, config):
        self.overlap_duty_cycle = (
                (self.overlap_duty_cycle * (config.duty_cycle_period - 1) + self.overlap) /
                config.duty_cycle_period
        )

        for segment in self.proximal_segments:
            for synapse in segment.synapses:
                synapse.permanence += config.permanence_inc * 0.1

        self.activity_duty_cycle = (
                (self.activity_duty_cycle * (config.duty_cycle_period - 1) + self.overlap) /
                config.duty_cycle_period
        )
        self.boost = math.exp(
            -config.boost_strength * (self.activity_duty_cycle - config.local_area_density)
        )

    def calculate_overlap(self, config):
        self.overlap = 0.0

        for segment in self.proximal_segments:
            for synapse in segment.synapses:
                if synapse.permanence >= config.permanence_threshold and synapse.is_active:
                    self.overlap += 1

        if self.overlap < config.segment_activation_threshold:
            self.overlap = 0.0
        else:
            self.overlap *= self.boost

    def mark_predictive_cells(self):
        for cell in self.cells:
            if cell.was_predictive:
                cell.active = True
                self.predicted = True

        if not self.predicted:
            for cell in self.cells:
                cell.active = True
            self.bursting = True

