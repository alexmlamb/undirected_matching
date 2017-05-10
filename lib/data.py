'''Module for handling data.

'''

import logging

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import SourcewiseTransformer, Transformer
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
        index = self.sources.index('features')
        p = self.amount
        x = np.lib.pad(
            batch[index], ((0, 0), (0, 0), (p, p), (p, p)), 'constant',
            constant_values=(0))
        batch = list(batch)
        batch[index] = x
        return tuple(batch)


class Rescale(Transformer):
    def __init__(self, data_stream, min=0, max=1, tanh=True, **kwargs):
        super(Rescale, self).__init__(data_stream=data_stream,
                                  produces_examples=False, **kwargs)
        self.min = min
        self.max = max
        self.tanh = tanh
    
    def transform_batch(self, batch):
        index = self.sources.index('features')
        x = float(self.max - self.min) * (batch[index] / 255.) - self.min
        if self.tanh:
            x = 2. * x - 1. 
        x = x.astype(floatX)
        batch = list(batch)
        batch[index] = x
        return tuple(batch)
    
    
class OneHotEncoding(SourcewiseTransformer):
    """Converts integer target variables to one hot encoding.

    It assumes that the targets are integer numbers from 0,... , N-1.
    Since it works on the fly the number of classes N needs to be
    specified.

    Batch input is assumed to be of shape (N,) or (N, 1).

    Parameters
    ----------
    data_stream : :class:`DataStream` or :class:`Transformer`.
        The data stream.
    num_classes : int
        The number of classes.

    """
    def __init__(self, data_stream, num_classes, **kwargs):
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(OneHotEncoding, self).__init__(
            data_stream, False, **kwargs)
        self.num_classes = num_classes

    def transform_batch(self, batch):
        index = self.sources.index('targets')
        labels = batch[index]
        
        if np.max(labels) >= self.num_classes:
            raise ValueError("all entries in source_batch must be lower than "
                             "num_classes ({})".format(self.num_classes))
        shape = labels.shape
        labels = labels.flatten()
        if shape[1] == 1:
            output = np.zeros((shape[0], self.num_classes),
                              dtype=floatX)
        else:
            output = np.zeros((shape[0] * shape[1], self.num_classes),
                              dtype=floatX)
        
        for i in range(self.num_classes):
            output[labels == i, i] = 1
        if shape[1] != 1:
            output = output.reshape((shape[0], shape[1], self.num_classes))

        batch = list(batch)
        batch[index] = output
        return tuple(batch)


def load_stream(batch_size=None, source=None, data_min=0, data_max=1,
                tanh=False):
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
            
        label_dataset = H5PYDataset(source, which_sets=(sets,),
                                    sources=['targets'], load_in_memory=True)
        labels, = label_dataset.data_sources
    
        for i, shape in enumerate(shapes):
            shape = list(shape)
            shape[0] = examples
            if i != 0 and shape[1] == 1: #HACK
                shape[1] = labels.max()+1
            shapes[i] = tuple(shape)
    
        if sets == 'test':
            batch_size = 256
        scheme = ShuffledScheme(examples=examples, batch_size=batch_size)
        stream = Rescale(DataStream(dataset, iteration_scheme=scheme),
                         min=data_min, max=data_max, tanh=tanh)
        stream = OneHotEncoding(stream, labels.max()+1)
        data_streams[sets] = stream
        data_shapes[sets] = shapes

    return data_streams, data_shapes