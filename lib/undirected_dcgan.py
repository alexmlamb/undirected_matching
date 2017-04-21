#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import logging
import sys
import os
from os import path
import time

import lasagne
from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm
from lasagne.nonlinearities import sigmoid, LeakyRectify, sigmoid
from lasagne.layers import (
    batch_norm, DenseLayer, InputLayer, ReshapeLayer)
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer  # override
from matplotlib import pyplot as plt
import numpy as np
from progressbar import Bar, ProgressBar, Percentage, Timer
import theano
import theano.tensor as T


DIM_X = 28
DIM_Y = 28
DIM_C = 1
lrelu = LeakyRectify(0.2)
floatX = lasagne.utils.floatX

# ##################### UTIL #####################

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.propagate = False
file_formatter = logging.Formatter(
    '%(asctime)s:%(name)s[%(levelname)s]:%(message)s')
stream_formatter = logging.Formatter(
    '[%(levelname)s:%(name)s]:%(message)s' + ' ' * 40)

def set_stream_logger(verbosity):
    global logger

    if verbosity == 0:
        level = logging.WARNING
        lstr = 'WARNING'
    elif verbosity == 1:
        level = logging.INFO
        lstr = 'INFO'
    elif verbosity == 2:
        level = logging.DEBUG
        lstr = 'DEBUG'
    else:
        level = logging.INFO
        lstr = 'INFO'
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.terminator = ''
    ch.setLevel(level)
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)
    logger.info('Setting logging to %s' % lstr)

def set_file_logger(file_path):
    global logger
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    fh.terminator = ''
    logger.info('Saving logs to %s' % file_path)
    
def update_dict_of_lists(d_to_update, **d):
    '''Updates a dict of list with kwargs.

    Args:
        d_to_update (dict): dictionary of lists.
        **d: keyword arguments to append.

    '''
    for k, v in d.iteritems():
        if k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]

# ################## DATA ##################

def load_mnist(self, source, mode='train'):
    '''Fetch data from gzip pickle.

    Args:
        source (str): path to source.
        mode (str): `train`, `test`, or `valid`.

    '''
    with gzip.open(source, 'rb') as f:
        x = cPickle.load(f)

    if mode == 'train':
        X = np.float32(x[0][0])
        Y = np.float32(x[0][1])
    elif mode == 'valid':
        X = np.float32(x[1][0])
        Y = np.float32(x[1][1])
    elif mode == 'test':
        X = np.float32(x[2][0])
        Y = np.float32(x[2][1])
    else:
        raise ValueError()

    return X, Y

def load_svhn(self, source, mode='train', greyscale=False):
    if mode == 'train':
        source_file = path.join(source, 'train_32x32.mat')
        data_dict = io.loadmat(source_file)
        X = data_dict['X']
        Y = data_dict['y']
    elif mode == 'valid':
        source_file = path.join(source, 'test_32x32.mat')
        data_dict = io.loadmat(source_file)
        X = data_dict['X']
        Y = data_dict['y']
    elif mode == 'test':
        source_file = path.join(source, 'test_32x32.mat')
        data_dict = io.loadmat(source_file)
        X = data_dict['X']
        Y = data_dict['y']
    else:
        raise ValueError()
    
    X = X.reshape((X.shape[0] * X.shape[1], X.shape[2], X.shape[3])).transpose(2, 1, 0)
    
    X_r = X[:, 2]
    X_g = X[:, 0]
    X_b = X[:, 1]
    
    if greyscale:
        X = 0.299 * X_r + 0.587 * X_b + 0.114 * X_g
    else:
        X = np.concatenate([X_r, X_b, X_g], axis=1)

    X = X - X.min(axis=0)
    X = X / X.max(axis=0)
    X = 2 * X - 1.

    return X.astype(floatX), Y.astype('int64')

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
##################### MODEL #######################

class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.Orthogonal(),
                                (self.input_shape[1], num_filters) + self.filter_size,
                                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                                (num_filters,),
                                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i * s - 2 * p + f - 1
                      for i, s, p, f in zip(input_shape[2:],
                                            self.stride,
                                            self.pad,
                                            self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)
    
    
class GaussianSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)
    

def build_decoder(Z=None, dim_z=128):
    layer = InputLayer(shape=(None, dim_z), input_var=Z)
    layer = batch_norm(DenseLayer(layer, 128 * DIM_X * DIM_Y // 16))
    layer = ReshapeLayer(layer, ([0], 128, DIM_X // 4, DIM_Y // 4))
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, pad=2))
    layer = Deconv2DLayer(layer, DIM_C, 5, stride=2, pad=2,
                          nonlinearity=DATA_NONLINEARITY)
    logger.debug('Generator output: {}'.format(layer.output_shape))
    return layer

def build_encoder(input_var=X, dim_z=128):
    layer = InputLayer(shape=(None, DIM_C, DIM_X, DIM_Y), input_var=X)
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad=2,
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad=2,
                                   nonlinearity=lrelu))
    logger.debug('Discriminator output: {}'.format(layer.output_shape))
    mu = DenseLayer(layer, dim_z, nonlinearity=None)
    log_sigma = DenseLayer(layer, dim_z, nonlinearity=None)
    return mu, log_sigma

def build_discriminator(X=None, Z=None, dim_z=128):
    layer_x = InputLayer(shape=(None, DIM_C, DIM_X, DIM_Y), input_var=X)
    layer_z = InputLayer(shape=(None, dim_z), input_var=X)
    layer_x = batch_norm(Conv2DLayer(layer_x, 64, 5, stride=2, pad=2,
                                     nonlinearity=lrelu))
    layer_x = batch_norm(Conv2DLayer(layer_x, 128, 5, stride=2, pad=2,
                                     nonlinearity=lrelu))
    layer_z = batch_norm(DenseLayer(layer_z, 128, nonlinearity=lrelu))
    layer = ConcatLayer([layer_x, layer_y])
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    layer = DenseLayer(layer, 1, nonlinearity=None)
    logger.debug('Discriminator output: {}'.format(layer.output_shape))
    return layer

# ############################# MATH ###############################

def log_sum_exp(x, axis=None):
    '''Numerically stable log( sum( exp(A) ) ).
    '''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis)
    return y

def norm_exp(log_factor):
    '''Gets normalized weights.
    '''
    log_factor = log_factor - T.log(log_factor.shape[0]).astype('float32')
    w_norm   = log_sum_exp(log_factor, axis=0)
    log_w    = log_factor - T.shape_padleft(w_norm)
    w_tilde  = T.exp(log_w)
    return w_tilde

# ############################# LOSSES ###############################

def BGAN(fake_out, real_out, log_Z, use_Z=True):    
    log_d1 = -T.nnet.softplus(-fake_out)
    log_d0 = -fake_out - T.nnet.softplus(-fake_out)
    log_w = log_d1 - log_d0

    log_N = T.log(log_w.shape[0]).astype(log_w.dtype)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est = theano.gradient.disconnected_grad(log_Z_est)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
        
    if use_Z:
        generator_loss = ((fake_out - log_Z) ** 2).mean()
    else:
        generator_loss = (fake_out ** 2).mean()
    discriminator_loss = T.nnet.softplus(-real_out).mean() + (
        T.nnet.softplus(-fake_out) + fake_out).mean()
    return generator_loss, discriminator_loss, log_w, w_tilde, log_Z_est

def GAN(fake_out, real_out):
    log_d1 = -T.nnet.softplus(-fake_out)
    log_d0 = -fake_out - T.nnet.softplus(-fake_out)
    log_w = log_d1 - log_d0

    log_N = T.log(log_w.shape[0]).astype(log_w.dtype)
    log_Z_est = log_sum_exp(log_w - log_N, axis=0)
    log_Z_est = theano.gradient.disconnected_grad(log_Z_est)
    log_w_tilde = log_w - T.shape_padleft(log_Z_est) - log_N
    w_tilde = T.exp(log_w_tilde)
         
    generator_loss = T.nnet.softplus(-fake_out).mean()
    discriminator_loss = T.nnet.softplus(-real_out).mean() + (
        T.nnet.softplus(-fake_out) + fake_out).mean()
    return generator_loss, discriminator_loss, log_w, w_tilde, log_Z_est

# ############################## MAIN ################################

def summarize(results, samples, image_dir=None, prefix=''):
    results = dict((k, np.mean(v)) for k, v in results.items())
    logger.info(results)
    if image_dir is not None:
        plt.imsave(path.join(image_dir, '{}.png'.format(prefix)),
                   (samples.reshape(10, 10, DIM_X, DIM_Y)
                    .transpose(0, 2, 1, 3)
                    .reshape(10 * DIM_X, 10 * DIM_Y)),
                   cmap='gray')

def main(num_epochs=None, method=None, batch_size=None,
         learning_rate=None, beta=None, n_steps=10,
         image_dir=None, binary_dir=None,
         prior=None, dim_z=None, source=None, archive=False, dataset='svhn'):
    
    # DATA
    if dataset == 'mnist':
        data = load_mnist(source)
    elif dataset == 'svhn':
        data = load_svhn(source)
    else:
        raise NotImplementedError(dataset)
    train_samples = data.shape[0]

    # VAR
    noise_var = T.matrix('noise')
    input_var = T.tensor4('inputs')
    log_Z = theano.shared(lasagne.utils.floatX(0.), name='log_Z')

    # MODEL
    logger.info('Building model and graph')
    decoder = None
    encoder = None
    gaussian_noise = None
    
    Z = noise_var
    X = None
    
    for i in xrange(n_steps):
        if decoder is None:
            decoder = build_decoder(Z, dim_z=dim_z)
            X = lasagne.get_output(decoder)
        else:
            X = lasagne.get_output(decoder, Z)
            
        if encoder is None:
            encoder = build_encoder(X, dim_z=dim_z)
            mu, log_sigma = lasagne.get_output(encoder)
        else:
            mu, log_sigma = lasagne.get_output(encoder, X)
            
        if gaussian_noise is None:
            gaussian_noise = GaussianSampleLayer(*encoder)
            Z = lasagne.get_output(gaussian_noise)
        else:
            Z = lasagne.get_output(gaussian_noise, [mu, log_sigma])
    
    mu, log_sigma = lasagne.get_output(encoder, input_var)
    Z_hat = lasagne.get_output(gaussian_noise, [mu, log_sigma])
    discriminator = build_discriminator(input_var, Z_hat)

    # GRAPH
    real_out = lasagne.layers.get_output(discriminator)
    fake_out = lasagne.layers.get_output(discriminator, {'X': X, 'Z': Z})

    if method == 'BGAN':
        generator_loss, discriminator_loss, log_w, w_tilde, log_Z_est = BGAN(
            fake_out, real_out, log_Z)
        other_loss, _, _, _, _ = GAN(fake_out, real_out)
    elif method == 'GAN':
        generator_loss, discriminator_loss, log_w, w_tilde, log_Z_est = GAN(
            fake_out, real_out)
        other_loss, _, _, _, _ = BGAN(fake_out, real_out, log_Z)
    else:
        raise NotImplementedError('Unsupported method `{}`'.format(method))

    # OPTIMIZER
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)
    discriminator_params = lasagne.layers.get_all_params(discriminator,
                                                         trainable=True)
    
    eta = theano.shared(lasagne.utils.floatX(learning_rate))
    
    updates = lasagne.updates.adam(
        generator_loss, generator_params, learning_rate=eta, beta1=beta)
    updates.update(lasagne.updates.adam(
        discriminator_loss, discriminator_params, learning_rate=eta, beta1=beta))
    updates.update([(log_Z, 0.95 * log_Z + 0.05 * log_Z_est.mean())])

    # COMPILE
    results = {
        'p(real)': (T.nnet.sigmoid(real_out) > .5).mean(),
        'p(fake': (T.nnet.sigmoid(fake_out) < .5).mean(),
        'G loss': generator_loss,
        'Other loss': other_loss,
        'D loss': discriminator_loss,
        'log Z': log_Z,
        'log Z est': log_Z_est.mean(),
        'log_Z est var': log_Z_est.std() ** 2,
        'log w': log_w.mean(),
        'log w var': log_w.std() ** 2,
        'norm w': w_tilde.mean(),
        'norm w var': w_tilde.std() ** 2,
        'ESS': (1. / (w_tilde ** 2).sum(0)).mean()
    }
    train_fn = theano.function([noise_var, input_var],
                               results,
                               updates=updates)

    gen_fn = theano.function(
        [noise_var], lasagne.layers.get_output(generator, deterministic=True))

    # TRAIN
    logger.info('Training...')
    
    results = {}
    for epoch in range(num_epochs):
        u = 0
        prefix = '{}_{}'.format(method, epoch)
        
        e_results = {}
        widgets = ['Epoch {}, '.format(epoch), Timer(), Bar()]
        pbar = ProgressBar(
            widgets=widgets, maxval=(train_samples // batch_size)).start()
        prefix = str(epoch)
        
        start_time = time.time()
        batch0 = None
        for batch in iterate_minibatches(data, batch_size,
                                         shuffle=True):
            inputs = batch
            if inputs.shape[0] == batch_size:
                if batch0 is None: batch0 = inputs
                
                if prior == 'uniform':
                    noise = floatX(np.random.rand(len(inputs), dim_z))
                elif prior == 'gaussian':
                    noise = floatX(numpy.random.normal(size=(len(inputs), dim_z)))
                    
                outs = train_fn(noise, inputs)
                outs = dict((k, np.asarray(v)) for k, v in outs.items())
                
                update_dict_of_lists(e_results, **outs)
                u += 1
                pbar.update(u)
            else:
                logger.error('Batch does not fit batch size. Skipping.')
            
        update_dict_of_lists(results, **e_results)
        np.savez(path.join(binary_dir, '{}_results.npz'.format(prefix)),
                 **results)
            
        try:
            if prior == 'uniform':
                noise = floatX(np.random.rand(100, dim_z))
            elif prior == 'gaussian':
                noise = floatX(numpy.random.normal(size=(64, dim_z)))
            samples = gen_fn(noise)
            summarize(results, samples, image_dir=image_dir,
                      prefix=prefix)
        except Exception as e:
            logger.error(e)
            pass

        logger.info('Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, num_epochs, time.time() - start_time))
        
        if not archive:
            prefix = method
        np.savez(path.join(binary_dir, '{}_generator_params.npz'.format(prefix)),
                 *lasagne.layers.get_all_param_values(generator))
        np.savez(path.join(binary_dir,
                           '{}_discriminator_params.npz'.format(prefix)),
                 *lasagne.layers.get_all_param_values(discriminator))
    
    # Load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


_defaults = dict(
    learning_rate=1e-3,
    beta=0.5,
    num_epochs=200,
    batch_size=64,
    method='BGAN',
    dim_z=128,
    prior='gaussian'
)

def make_argument_parser():
    '''Generic experiment parser.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    parser.add_argument('-a', '--archive', action='store_true', default=False)
    return parser

def setup_out_dir(out_path, name=None):
    if out_path is None:
        raise ValueError('Please set `--out_path` (`-o`) argument.')
    if name is not None:
        out_path = path.join(out_path, name)
        
    binary_dir = path.join(out_path, 'binaries')
    image_dir = path.join(out_path, 'images')
    if not path.isdir(out_path):
        logger.info('Creating out path `{}`'.format(out_path))
        os.mkdir(out_path)
        os.mkdir(binary_dir)
        os.mkdir(image_dir)
        
    logger.info('Setting out path to `{}`'.format(out_path))
    logger.info('Logging to `{}`'.format(path.join(out_path, 'out.log')))
    set_file_logger(path.join(out_path, 'out.log'))
        
    return dict(binary_dir=binary_dir, image_dir=image_dir)

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    set_stream_logger(args.verbosity)
    out_paths = setup_out_dir(args.out_path, args.name)
    
    kwargs = dict()
    kwargs.update(**_defaults)
    kwargs.update(out_paths)
    kwargs['archive'] = args.archive
    logger.info('kwargs: {}'.format(kwargs))
        
    main(**kwargs)
