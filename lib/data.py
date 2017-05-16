'''Module for handling data.

'''

import logging
import random

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
    def __init__(self, data_stream, min=0, max=1, tanh=True, rescale=True,
                 **kwargs):
        super(Rescale, self).__init__(data_stream=data_stream,
                                  produces_examples=False, **kwargs)
        self.min = min
        self.max = max
        self.tanh = tanh
        self.rescale = rescale
    
    def transform_batch(self, batch):
        index = self.sources.index('features')
        x = batch[index]
        if self.rescale:
            x = float(self.max - self.min) * (x / 255.) - self.min
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
    def __init__(self, data_stream, num_classes, max_length=None, **kwargs):
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(OneHotEncoding, self).__init__(
            data_stream, False, **kwargs)
        self.num_classes = num_classes
        self.max_length = max_length

    def transform_batch(self, batch):
        try:
            index = self.sources.index('targets')
            labels = batch[index]
            
        except ValueError:
            index = self.sources.index('captions')
            idx = random.randint(0, 9)
            labels = batch[index][:, idx, :self.max_length]
            
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
                tanh=False, max_length=None):
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
        try:
            label_dataset = H5PYDataset(source, which_sets=(sets,),
                                        sources=['targets'],
                                        load_in_memory=True)
        except:
            label_dataset = H5PYDataset(source, which_sets=(sets,),
                                        sources=['captions'],
                                        load_in_memory=True)
        labels, = label_dataset.data_sources
    
        if sets == 'test':
            batch_size = 256
        scheme = ShuffledScheme(examples=examples, batch_size=batch_size)
        stream = Rescale(DataStream(dataset, iteration_scheme=scheme),
                         min=data_min, max=data_max, tanh=tanh)
        stream = OneHotEncoding(stream, labels.max()+1, max_length=max_length)
        data = stream.get_epoch_iterator().next()
        shapes = dict((k, d.shape) for k, d in zip(stream.which_sources, data))
        
        for k, shape in shapes.items():
            shape = list(shape)
            shape[0] = examples
            shapes[k] = tuple(shape)
    
        data_streams[sets] = stream
        data_shapes[sets] = shapes

    return data_streams, data_shapes


def prepare_data(source, pad_to=None, batch_size=None, model_args=None,
                 **kwargs):
    logger.info('Perparing data from `{}`'.format(source))
    
    if source is None: raise ValueError('Source must be provided.')
    
    datasets, data_shapes = load_stream(source=source, batch_size=batch_size)
    if pad_to is not None:
        logger.info('Padding data to {}'.format(pad_to))
        for k in datasets.keys():
            data_shape = data_shapes[k][0]
            dataset = datasets[k]
            x_dim, y_dim = data_shape[2], data_shape[3]
            p = (pad_to[0] - x_dim) // 2 # Some bugs could occur here.
            datasets[k] = Pad(dataset, p)
            data_shape = (data_shape[0], data_shape[1], pad_to[0], pad_to[1])
            data_shapes[k] = tuple([data_shape] + list(data_shapes[k])[1:])
            
    _, dim_c, dim_x, dim_y = data_shapes['train']['features']
    logger.info(
        'Setting dim_x to {}, dim_y to {}, and dim_c to {}'.format(
            dim_x, dim_y, dim_c))
    if 'targets' in data_shapes['train'].keys():
        shape = data_shapes['train']['targets']
        if len(shape) == 2:
            _, dim_l = shape
            dim_w = 1
        else:
            _, dim_w, dim_l = shape
        logger.info('Setting dim_w to {} and dim_l to {}'.format(
            dim_w, dim_l))
    elif 'captions' in data_shapes['train'].keys():
        _, dim_w, dim_l = data_shapes['train']['captions']
        logger.info('Setting dim_w to {} and dim_l to {}'.format(
            dim_w, dim_l))
        
    if model_args is not None:
        model_args.update(dim_x=dim_x, dim_y=dim_y, dim_c=dim_c, dim_w=dim_w,
                          dim_l=dim_l)
        
    return datasets, data_shapes