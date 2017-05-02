'''Module for handling data.

'''

import logging

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer


logger = logging.getLogger('UDGAN.data')


class Pad(Transformer):
    def __init__(self, data_stream, **kwargs):
        super(Pad, self).__init__(data_stream=data_stream,
                                  produces_examples=False, **kwargs)
    
    def transform_batch(self, batch):
        batch[0] = np.lib.pad(
            batch[0], ((0, 0), (0, 0), (2, 2), (2, 2)), 'constant',
            constant_values=(0))
        return batch


def load_stream(batch_size=None, source=None):
    logger.info('Loading data from `{}`'.format(source))
    
    data_streams = {}
    data_shapes = {}
    for sets in ['train', 'test']:
        dataset = H5PYDataset(source, which_sets=(sets,))
        logger.debug('Sources are {}.'.format(dataset.sources))
        logger.debug('Axis labels are {}.'.format(dataset.axis_labels))
        logger.debug('Dataset contains {} examples.'.format(
            dataset.num_examples))
        
        examples = dataset.num_examples
        handle = dataset.open()
        data = dataset.get_data(handle, slice(0, 10))
        shapes = [d.shape for d in data]
        for i, shape in enumerate(shapes):
            shape = list(shape)
            shape[0] = examples
            shapes[i] = tuple(shape)
    
        scheme = ShuffledScheme(examples=examples, batch_size=batch_size)
        stream = DataStream(dataset, iteration_scheme=scheme)
        data_streams[sets] = stream
        data_shapes[sets] = shapes

    return data_streams, data_shapes