'''Module for handling data.

'''

import logging

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer
import numpy as np
import theano


floatX = theano.config.floatX
logger = logging.getLogger('UDGAN.data')


class Pad(Transformer):
    def __init__(self, data_stream, amount, **kwargs):
        super(Pad, self).__init__(data_stream=data_stream,
                                  produces_examples=False, **kwargs)
        self.amount = amount
    
    def transform_batch(self, batch):
        p = self.amount
        x = np.lib.pad(
            batch[0], ((0, 0), (0, 0), (p, p), (p, p)), 'constant',
            constant_values=(0))
        batch = list(batch)
        batch[0] = x
        return tuple(batch)


class Rescale(Transformer):
    def __init__(self, data_stream, min=0, max=1, **kwargs):
        super(Rescale, self).__init__(data_stream=data_stream,
                                  produces_examples=False, **kwargs)
        self.min = min
        self.max = max
    
    def transform_batch(self, batch):
        x = float(self.max - self.min) * (batch[0] / 255.) - self.min
        x = x.astype(floatX)
        batch = list(batch)
        batch[0] = x
        return tuple(batch)


def load_stream(batch_size=None, source=None, data_min=0, data_max=1):
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
        stream = Rescale(DataStream(dataset, iteration_scheme=scheme),
                         min=data_min, max=data_max)
        data_streams[sets] = stream
        data_shapes[sets] = shapes

    return data_streams, data_shapes