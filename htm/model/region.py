import math
import uuid
from functools import reduce
import random

from htm.model.column import Column
from htm.model.segment import DistalSegment
from htm.model.synapse import DistalSynapse


class RegionConfig:
    def __init__(
            self,
            active_columns_num: int,
            shape: tuple,
            permanence_inc: float,
            permanence_dec: float,
            distal_permanence_inc: float,
            distal_permanence_dec: float,
            initial_permanence: float,
            boost_inc: float,
            overlap_threshold: float,
            permanence_threshold: float,
            segment_activation_threshold: int,
            segment_min_threshold: int,
            cells_per_column: int,
            segments_per_cell: int,
            synapses_per_segment_limit: int,
            initial_synapses_per_segment: int,
            permanence_limit: float,
            decay: float,
            learning: bool,
            boost_strength: float,
            max_new_synapses: int,
            duty_cycle_period: int = 100
    ):
        self.active_columns_num = active_columns_num
        self.shape = shape
        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.distal_permanence_inc = distal_permanence_inc
        self.distal_permanence_dec = distal_permanence_dec
        self.boost_inc = boost_inc
        self.permanence_threshold = permanence_threshold
        self.overlap_threshold = overlap_threshold
        self.segment_activation_threshold = segment_activation_threshold
        self.segment_min_threshold = segment_min_threshold
        self.learning = learning
        self.cells_per_column = cells_per_column
        self.segments_per_cell = segments_per_cell
        self.synapses_per_segment_limit = synapses_per_segment_limit
        self.permanence_limit = permanence_limit
        self.decay = decay
        self.boost_strength = boost_strength
        self.duty_cycle_period = duty_cycle_period
        self.max_new_synapses = max_new_synapses
        self.initial_permanence = initial_permanence
        self.initial_synapses_per_segment = initial_synapses_per_segment

    @property
    def local_area_density(self):
        return self.active_columns_num / reduce(lambda a, b: a * b, self.shape, 1)


class Region:
    def __init__(
            self,
            config: RegionConfig
    ):
        self.id = uuid.uuid4().int
        self.config = config

        n_columns = reduce(lambda a, b: a * b, config.shape, 1)

        self.columns = [Column(config.cells_per_column) for _ in range(n_columns)]
        self.cells_ids = set()
        self.cells = dict()
        self.active_columns = []
        self.predicted_cells = []
        self.bursting_cells = []
        self.learning_segments = []

        for column in self.columns:
            self.cells_ids.update(column.cells_ids)

            for cell in column.cells:
                self.cells[cell.id] = cell

    def initialize_distal_segments(self):
        cells_ids = list(self.cells_ids)

        for column in self.columns:
            for cell in column.cells:

                cell.distal_segments = [DistalSegment(cell) for _ in range(self.config.segments_per_cell)]

                for segment in cell.distal_segments:
                    presynaptic_cells = random.choices(cells_ids, k=self.config.initial_synapses_per_segment)

                    for cell_id in presynaptic_cells:
                        presynaptic_cell = self.cells[cell_id]
                        synapse = DistalSynapse(presynaptic_cell, segment)
                        synapse.permanence = random.randrange(0.0, self.config.permanence_limit)
                        segment.synapses.append(synapse)

    def spatial_pooling(self):
        self._step()
        self._spatial_phase_1()
        self._spatial_phase_2()
        self._spatial_phase_3()

    def temporal_pooling(self):
        self._temporal_phase_1()
        self._temporal_phase_2()
        self._temporal_phase_3()
        self._temporal_phase_4()

    def _step(self):
        self.active_columns = []
        self.predicted_cells = []
        self.bursting_cells = []
        self.learning_segments = []

        for column in self.columns:
            column.step()

    def _spatial_phase_1(self):
        for column in self.columns:
            column.calculate_overlap(self.config)

    def _spatial_phase_2(self):
        top_active_columns = list(sorted(self.columns, key=lambda c: -c.overlap))[0:self.config.active_columns_num]

        for column in self.columns:
            if column in top_active_columns:
                column.active = True
                self.active_columns.append(column)

            else:
                column.active = False
                column.overlap = 0.0

    def _spatial_phase_3(self):
        minimal_odc = self._minimal_odc

        for column in self.columns:
            for segment in column.proximal_segments:
                segment.increase_active_permanences(self.config.permanence_inc, self.config.permanence_limit)
                segment.decrease_inactive_permanences(self.config.permanence_dec)

            column.update_duty_cycles(self.config)
            minimal_odc = min(column.overlap_duty_cycle, minimal_odc)

    def _temporal_phase_1(self):
        for column in self.active_columns:
            for cell in column.cells:
                if cell.was_predictive:
                    cell.active = True
                    column.predicted = True
                    self.predicted_cells.append(cell)

            if not column.predicted:
                column.bursting = True

                for cell in column.cells:
                    cell.active = True
                    self.bursting_cells.append(cell)

    def _temporal_phase_2(self):
        for cell in self.predicted_cells:
            for segment in cell.distal_segments:
                if segment.was_active:
                    segment.learning = True
                    self.learning_segments.append(segment)

        for cell in self.bursting_cells:
            segment = cell.best_segment(self.config)

            if segment:
                segment.learning = True

                self.learning_segments.append(segment)

            elif len(cell.distal_segments) < self.config.segments_per_cell:
                segment = DistalSegment(cell)
                cell.distal_segments.append(segment)
                segment.learning = True

                self.learning_segments.append(segment)

    def _temporal_phase_3(self):
        previously_active = {cell for cell in self.cells.values() if cell.was_active}

        for segment in self.learning_segments:
            for synapse in segment.synapses:
                if synapse.input_cell.was_active:
                    synapse.permanence = min(
                        self.config.permanence_limit,
                        synapse.permanence + self.config.distal_permanence_inc
                    )
                else:
                    synapse.permanence = max(
                        0.0,
                        synapse.permanence - self.config.distal_permanence_dec
                    )

            segment.synapses = [synapse for synapse in segment.synapses if synapse.permanence > 0]

            already_sampled = {synapse.input_cell for synapse in segment.synapses}

            candidates = previously_active - already_sampled
            candidates = random.choices(
                list(candidates), k=min(
                    len(candidates),
                    self.config.max_new_synapses,
                    self.config.synapses_per_segment_limit - len(segment.synapses)
                )
            )

            for cell in candidates:
                synapse = DistalSynapse(cell, segment)
                synapse.permanence = self.config.initial_permanence
                segment.synapses.append(synapse)

    def _temporal_phase_4(self):
        for column in self.columns:
            for cell in column.cells:
                for segment in cell.distal_segments:
                    overlap = 0

                    for synapse in segment.synapses:
                        if synapse.active:
                            overlap += 1

                    if overlap > self.config.segment_activation_threshold:
                        segment.active = True
                        cell.predictive = True

    @property
    def _minimal_odc(self):
        minimal = self.columns[0].overlap_duty_cycle

        for column in self.columns:
            if column.overlap_duty_cycle < minimal:
                minimal = column.overlap_duty_cycle

        return minimal
