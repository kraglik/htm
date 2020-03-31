import time
import numpy as np

from htm.decoders.float_decoder import FloatDecoder
from htm.model import InputLayer
from htm.model.connection import Connection, RandomConnectomeGenerator
from htm.model.region import *
from htm.encoders import *


def main():
    config = RegionConfig(
        active_columns_num=22,
        shape=(32, 32),
        segments_per_cell=10,
        synapses_per_segment_limit=20,
        initial_synapses_per_segment=6,
        segment_activation_threshold=6,
        segment_min_threshold=4,
        max_new_synapses=6,
        boost_inc=0.1,
        cells_per_column=8,
        learning=True,
        overlap_threshold=18.0,
        permanence_dec=0.1,
        permanence_inc=0.1,
        distal_permanence_dec=0.1,
        distal_permanence_inc=0.1,
        permanence_threshold=1.0,
        initial_permanence=1.2,
        permanence_limit=4.0,
        decay=1e-6,
        boost_strength=3.0,
    )

    input_layer = InputLayer((110,))
    region = Region(config)
    input_to_region = Connection(input_layer, region, RandomConnectomeGenerator(0.45))
    region_decoder = FloatDecoder(region, steps_ahead=1)
    float_enc = FloatEncoder(w=10, n=110, minimum=0.0, maximum=1.0)

    values = [i * 0.1 for i in range(11)]

    synapses_count = 0
    for column in region.columns:
        for cell in column.cells:
            for segment in cell.distal_segments:
                synapses_count += len(segment.synapses)

    print("synapses:", synapses_count)

    previous_prediction = 0.0

    for iteration in range(100):
        for value in values:
            input_layer.set_value(float_enc(value))
            print('spatial pooling step')
            region.spatial_pooling()
            print('temporal pooling step')
            region.temporal_pooling()
            print('prediction:', previous_prediction, "value:", value, "difference:", abs(value - previous_prediction))
            previous_prediction = region_decoder(value)

            synapses_count = 0
            for column in region.columns:
                for cell in column.cells:
                    for segment in cell.distal_segments:
                        synapses_count += len(segment.synapses)

            print(
                "synapses:", synapses_count,
                "predicted cells:", len(region.predicted_cells),
                "bursting cells:", len(region.bursting_cells)
            )


if __name__ == '__main__':
    main()
