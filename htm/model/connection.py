import random
from functools import reduce

from htm.model.column import Column
from htm.model.segment import ProximalSegment
from htm.model.synapse import ProximalSynapse


class Connection:
    def __init__(
            self,
            underlying_region,
            overlying_region,
            connectome_generator
    ):
        self.underlying = underlying_region
        self.overlying = overlying_region
        self.connectome_generator = connectome_generator

        for column in overlying_region.columns:
            coords = self.connectome_generator(underlying_region.shape, overlying_region)

            proximal_segment = ProximalSegment()

            proximal_synapses = [
                ProximalSynapse(
                    underlying_region,
                    tuple(coord),
                    proximal_segment,
                    random.randrange(
                        self.overlying.config.permanence_threshold,
                        self.overlying.config.permanence_limit
                    )
                )
                for coord in coords
            ]

            proximal_segment.synapses = proximal_synapses
            column.proximal_segments.append(proximal_segment)


class ApicalConnection:
    def __init__(
            self,
            underlying_layer,
            overlying_layer,
            connectome_generator
    ):
        self.underlying = underlying_layer
        self.overlying = overlying_layer
        self.connectome_generator = connectome_generator


class RandomConnectomeGenerator:
    def __init__(self, connection_percent: float):
        self.connection_percent = connection_percent

    def __call__(self, input_shape, region):
        created_coords = set()

        connections_required = int(reduce(lambda a, b: a * b, input_shape, 1) * self.connection_percent)
        connections_required = max(10, connections_required)

        while len(created_coords) < connections_required:
            coord = (random.randint(0, dim - 1) for dim in input_shape)

            if coord not in created_coords:
                created_coords.add(coord)

        return created_coords


class TopologicalNearestConnectomeGenerator:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class RandomApicalConnectomeGenerator:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class TopologicalApicalNearestConnectomeGenerator:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass