#!/usr/bin/env python

'''Convolutional undirected GAN.

.. note::
   Initially make z_to_x and x_to_z fairly shallow networks.  Inject noise?fdfsj
   Use the fflayer class?

'''

import gzip
import cPickle as pickle
import random
import os
from os import path
import sys
#sys.setrecursionlimit(250)
import time

import imageio
import lasagne
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
from progressbar import Bar, ProgressBar, Percentage, Timer
import theano
import theano.tensor as T
from viz import plot_images

logger = logging.getLogger('UDGAN')

from data import load_stream, Pad
from exptools import make_argument_parser, setup_out_dir
from loggers import set_stream_logger
from loss import (accuracy, bgan_loss, bgan_loss_2, crossent, lsgan_loss,
                  wgan_loss)
from models import conv1, conv2, conv_ss_1, conv_ss_2
from utils import (
    init_tparams, join2, srng, dropout, inverse_sigmoid, join3, merge_images,
    sample_multinomial)


floatX = theano.config.floatX

try:
    SLURM_NAME = os.environ["SLURM_JOB_ID"]
except:
    SLURM_NAME = 'NA'
    
DIM_X = None
DIM_Y = None
DIM_C = None
DIM_Z = None
DIM_L = None
R_NONLINEARITY = None
MODULE = None

consider_constant = theano.gradient.disconnected_grad # changed from orginal


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

# PRETTY -----------------------------------------------------------------------

try:
    _, _columns = os.popen('stty size', 'r').read().split()
    _columns = int(_columns)
except ValueError:
    _columns = 1

def print_section(s):
    '''For printing sections to scripts nicely.

    Args:
        s (str): string of section

    '''
    h = s + ('-' * (_columns - len(s)))
    print h

# OPTIMIZER --------------------------------------------------------------------

def set_optimizer(dloss, gloss, dparams, gparams,
                  optimizer=None, op_args=None):
    '''Sets the loss and optimizer and gets updates
    
    '''
    op_args = op_args or {}
    
    logger.info("Setting optimizer. Using {} with args {}".format(
        optimizer, op_args))
    
    if optimizer == 'rmsprop':
        dupdates = lasagne.updates.rmsprop(dloss, dparams.values(), **op_args)
        gcupdates = lasagne.updates.rmsprop(gloss, gparams.values(), **op_args)
    elif optimizer == 'adam':
        dupdates = lasagne.updates.adam(dloss, dparams.values(), **op_args)
        gcupdates = lasagne.updates.adam(gloss, gparams.values(), **op_args)
    else:
        raise NotImplementedError(optimizer)

    dgcupdates = dupdates.copy()
    dgcupdates.update(gcupdates)
    
    return dgcupdates
    
# COMPILE ----------------------------------------------------------------------
    
def compile_train(updates, inputs, outputs):
    logger.info("Compiling train function")
    f = theano.function(
        inputs=inputs, outputs=outputs, updates=updates,
        on_unused_input='ignore')
    
    return f


def compile_generation(z, x, y):
    logger.info("Compiling generation function")
    f = theano.function([z], [x, y])
    return f


def compile_chain(z, gparams, num_steps_long=None, **model_args):
    logger.info("Compiling chain function")
    xs, ys, zs = p_chain(
        gparams, z, num_steps_long, **model_args)
    
    f = theano.function([z], xs + ys)
    
    return f


def compile_chain_x(x, gparams, num_steps_long=None, **model_args):
    logger.info("Compiling chain starting at x function")
    xs, ys, zs = p_chain(
        gparams, None, num_steps_long, x=x, **model_args)
    
    f = theano.function([x], xs + ys)
    
    return f


def compile_inpaint_chain(x, z, gparams, num_steps_long=None,
                          noise_damping=None, **model_args):
    logger.info("Compiling inpaint function")
    xs, ys = inpaint_chain(
        gparams, x, z, num_steps_long, noise_damping=noise_damping,
        **model_args)
    
    f = theano.function([x, z], xs + ys)
    
    return f


def compile_inpaint_x_chain(y, z, gparams, num_steps_long=None,
                            noise_damping=None, **model_args):
    logger.info("Compiling inpaint image from label function")
    xs = inpaint_images(
        gparams, y, z, num_steps_long, noise_damping=noise_damping,
        **model_args)
    
    f = theano.function([y, z], xs)
    
    return f


def compile_inpaint_y_chain(x, z, gparams, num_steps_long=None,
                            noise_damping=None, **model_args):
    logger.info("Compiling inpaint label from image function")
    ys = inpaint_labels(
        gparams, x, z, num_steps_long, noise_damping=noise_damping,
        **model_args)
    
    f = theano.function([x, z], ys)
    
    return f


def compile_piecewise_chain(z, x, gparams, **model_args):
    logger.info("Compiling piecewise chain function")
    f_z_to_x = theano.function([z], outputs=onestep_z_to_x(gparams, z))
    f_x_to_z = theano.function([x], outputs=onestep_x_to_z(gparams, x))
    
    return f_z_to_x, f_x_to_z
    
# VIZ --------------------------------------------------------------------------
    
def visualizer(num_steps_long=None):
    '''For eval, not for training.
    
    TODO
    
    '''
    pass

# MODELS -----------------------------------------------------------------------

def p_chain(p, z, num_iterations, pd_steps=None, x=None,
            y=None, test=False, **model_args):
    '''P chain
    
    .. note::
        the returned ylst are pre-sampling. This was done to do multiple
        samples later
    
    '''
    z_to_x = MODULE.z_to_x
    x_to_z = MODULE.x_to_z

    zlst = [z]
    plst = []
    xlst = []
    
    if x is not None:
        z = x_to_z(p, x, **model_args)
        xlst.append(x)
        zlst.append(z)

    for i in xrange(num_iterations):
        out = z_to_x(p, z, **model_args)
        
        if MODULE._semi_supervised:
            x, p_ = out
            plst.append(p_)
            y = sample_multinomial(p_)
        else:
            x = out
            
        xlst.append(x)

        if i < num_iterations - 1:
            if MODULE._semi_supervised:
                if i == num_iterations - pd_steps - 1:
                    # Changed this
                    z = x_to_z(p, consider_constant(x), y, **model_args) 
                else:
                    z = x_to_z(p, x, y, **model_args)
            else:
                if i == num_iterations - pd_steps - 1:
                    # Changed this
                    z = x_to_z(p, consider_constant(x), **model_args) 
                else:
                    z = x_to_z(p, x, **model_args)
            
            zlst.append(z)

    #for j in range(len(xlst)):
    #    xlst[j] = T.nnet.sigmoid(xlst[j])
    
    assert len(xlst) == len(zlst)
    return xlst, plst, zlst


def q_chain(p, x, y, num_iterations, test=False, **model_args):
    x_to_z = MODULE.x_to_z
    xlst = [x]
    ylst = [y]
    zlst = []
    
    #x_ = inverse_sigmoid(x)
    x_ = x
    
    if MODULE._semi_supervised:
        new_z = x_to_z(p, x_, y, **model_args)
    else:
        new_z = x_to_z(p, x_, **model_args)
        
    zlst.append(new_z)

    return xlst, ylst, zlst


def inpaint_chain(p, x, z, num_iterations, noise_damping=None, **model_args):
    z_to_x = MODULE.z_to_x
    x_to_z = MODULE.x_to_z
    #x_gt = inverse_sigmoid(x)
    x_gt = x
    xlst = []
    ylst = []
    sigma = 1.
    
    for i in xrange(num_iterations):
        out = z_to_x(p, z, noise_scale=sigma, **model_args)
        
        if MODULE._semi_supervised:
            x, p_ = out
            y = sample_multinomial(p_)
            ylst.append(y)
        else:
            x = out
            
        x = T.set_subtensor(x[:, :, :DIM_X // 2, :], x_gt[:, :, :DIM_X // 2, :])
        xlst.append(x)
        
        if MODULE._semi_supervised:
            z = x_to_z(p, x, y, noise_scale=sigma, **model_args)
        else:
            z = x_to_z(p, x, noise_scale=sigma, **model_args)
        if noise_damping is not None:
            sigma *= noise_damping

    #for j in range(len(xlst)):
    #    xlst[j] = T.nnet.sigmoid(xlst[j])
        
    return xlst, ylst


def inpaint_labels(p, x, z, num_iterations, **model_args):
    if not MODULE._semi_supervised:
        return None
    z_to_x = MODULE.z_to_x
    x_to_z = MODULE.x_to_z
    #x_gt = inverse_sigmoid(x)
    x_gt = x
    ylst = []
    
    for i in xrange(num_iterations):
        x, p_ = z_to_x(p, z, **model_args)
        y = sample_multinomial(p_)
        ylst.append(y)
        z = x_to_z(p, x_gt, y, **model_args)
        
    return ylst


def inpaint_images(p, y, z, num_iterations, **model_args):
    if not MODULE._semi_supervised:
        return None
    z_to_x = MODULE.z_to_x
    x_to_z = MODULE.x_to_z
    y_gt = y
    xlst = []
    
    for i in xrange(num_iterations):
        x, p_ = z_to_x(p, z, **model_args)
        xlst.append(x)
        z = x_to_z(p, x, y_gt, **model_args)
        
    return xlst


def onestep_z_to_x(p, z, **model_args):
    out = MODULE.z_to_x(p, z, **model_args)
    if MODULE._semi_supervised:
        x, y = out
    else:
        x = out
    #x = T.nnet.sigmoid(x)
    return x


def onestep_x_to_z(p, x, y, **model_args):
    #x = inverse_sigmoid(x)
    if MODULE._semi_supervised:
        new_z = x_to_z(p, x, y, **model_args)
    else:
        new_z = x_to_z(p, x, **model_args)
    return new_z


def make_model(num_steps=None, pd_steps=None, loss=None,
               n_samples=None, start_on_x=None, **model_args):
    '''Form the model and graph.
    
    '''
    
    if MODULE._semi_supervised and loss != 'bgan':
        raise NotImplementedError('Currently, semi-supervised is only '
                                  'implemented with BGAN (got {})'.format(
                                    loss))
    
    logger.info("Initializing parameters from {}".format(MODULE.__name__))
    gparams = MODULE.init_gparams({}, **model_args)
    dparams = MODULE.init_dparams({}, **model_args)

    logger.info("Setting input variables")
    x_in = T.tensor4('x_in')
    z_in = T.matrix('z_in')
    y_in = T.matrix('y_in')

    logger.info("Building graph")    
    if start_on_x:
        #x = inverse_sigmoid(x_in)
        x = x_in
        logger.info('Starting chain at data.')
        p_lst_x, p_lst_p, p_lst_z = p_chain(gparams, z_in, num_steps,
                                            pd_steps=pd_steps,
                                            x=x,
                                            **model_args)
        p_lst_x_, _, _ = p_chain(gparams, z_in, num_steps,
                                 pd_steps=pd_steps, **model_args)
    else:
        logger.info('Starting chain at gaussian noise.')
        outs = p_chain(gparams, z_in, num_steps, pd_steps=pd_steps,
                       **model_args)
        p_lst_x, p_lst_p, p_lst_z = outs
        plst_x_ = p_lst_x

    # p_list_p is pre-activation and sampling
    
    q_lst_x, q_lst_y, q_lst_z = q_chain(gparams, x_in, y_in, num_steps,
                                        **model_args)

    z_inf = q_lst_z[-1]

    logger.debug("p chain x: {}".format(p_lst_x))
    logger.debug("p chain p: {}".format(p_lst_p))
    logger.debug("p chain z: {}".format(p_lst_z))
    logger.debug("q chain x: {}".format(q_lst_x))
    logger.debug("q chain y: {}".format(q_lst_y))
    logger.debug("q chain z: {}".format(q_lst_z))

    logger.info('Using {} steps of p in discriminator'.format(pd_steps))
    D_p_lst = []
    if MODULE._semi_supervised:
        p_lst_p_e = [T.tile(p_[None, :, :], (n_samples, 1, 1))
                    for p_ in p_lst_p]
        p_lst_p_r = [p_.reshape((-1, DIM_L)) for p_ in p_lst_p_e]
        
        p_lst_x_e = [T.tile(x[None, :, :, :, :], (n_samples, 1, 1, 1, 1))
                     for x in p_lst_x]
        p_lst_z_e = [T.tile(z[None, :, :], (n_samples, 1, 1))
                     for z in p_lst_z]

        p_lst_x_r = [x.reshape((-1, DIM_C, DIM_X, DIM_Y))
                     for x in p_lst_x_e]
        p_lst_y_r = [sample_multinomial(p_) for p_ in p_lst_p_r]
        p_lst_y = [y.reshape((n_samples, -1, DIM_L)) for y in p_lst_y_r]
        p_lst_z_r = [z.reshape((-1, DIM_Z))
                     for z in p_lst_z_e]
        
        D_q_lst, D_feat_q = MODULE.discriminator(
            dparams, q_lst_x[-1], q_lst_y[-1], q_lst_z[-1], **model_args)
        
        dloss = 0.
        gloss = 0.
        for i in xrange(pd_steps):
            D_p_lst_, D_feat_p = MODULE.discriminator(
                dparams, p_lst_x_r[-(i + 1)], p_lst_y_r[-(i + 1)],
                p_lst_z_r[-(i + 1)], **model_args)
            D_p_lst = [D.reshape((n_samples, x_in.shape[0], -1))
                       for D in D_p_lst_]
        
            dloss_, gloss_ = bgan_loss_2(D_q_lst, D_p_lst,
                                         p_lst_y[-(i + 1)],
                                         p_lst_p[-(i + 1)])
            dloss += dloss_
            gloss += gloss_
            
    else:
        for i in xrange(pd_steps):
            D_p_lst_, D_feat_p = MODULE.discriminator(
                dparams, p_lst_x[-(i + 1)], p_lst_z[-(i + 1)], **model_args)
            D_p_lst += D_p_lst_
        D_q_lst, D_feat_q = MODULE.discriminator(
            dparams, q_lst_x[-1], q_lst_z[-1], **model_args)
        
        if loss == 'lsgan':
            loss_fn = lsgan_loss
        elif loss == 'wgan':
            loss_fn = wgan_loss
            logger.warn('Lipschitz constraint not implemented, so WGAN will not '
                        'work correctly.')
        elif loss == 'bgan':
            loss_fn = bgan_loss
        else:
            raise NotImplementedError(loss)
        
        dloss, gloss = loss_fn(D_q_lst, D_p_lst)
    
    results = {
        'D loss': dloss,
        'G loss': gloss,
        'p(real)': (q_lst_x[-1] > 0.5).mean(),
        'p(fake)': (p_lst_x[-1] < 0.5).mean()
    }

    if MODULE._semi_supervised:
        inputs = [x_in, z_in, y_in]
    else:
        inputs = [x_in, z_in]

    return dloss, gloss, dparams, gparams, inputs, results, p_lst_x, p_lst_p

# DATA -------------------------------------------------------------------------

def prepare_data(source, pad_to=None, batch_size=None, tanh=None, **kwargs):
    logger.info('Perparing data from `{}`'.format(source))
    global DIM_X, DIM_Y, DIM_C, DIM_L
    
    if source is None: raise ValueError('Source must be provided.')
    
    datasets, data_shapes = load_stream(source=source, batch_size=batch_size,
                                        tanh=tanh)
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

    shape = data_shapes['train'][0]
    
    logger.info('Setting DIM_X to {}, DIM_Y to {}, and DIM_C to {}'.format(
        shape[2], shape[3], shape[1]))

    DIM_X = shape[2]
    DIM_Y = shape[3]
    DIM_C = shape[1]
    
    shape = data_shapes['train'][1]
    logger.info('Setting DIM_L to {}'.format(shape[1]))
    DIM_L = shape[1]
    
    return datasets, data_shapes

# TRAIN ------------------------------------------------------------------------

def train(train_fn, gen_fn,
          chain_fn, chain_fn_x,
          inpaint_fn, inpaint_fn_d, inpaint_fn_x, inpaint_fn_y,
          save_fn, datasets, data_shapes, persist_p_chain=None,
          latent_sparse=None, num_epochs=None, blending_rate=None,
          latent_sparse_num=None, batch_size=None, binary_dir=None,
          image_dir=None, archive=None):
    '''Train method.
    
    '''
    
    train_samples = data_shapes['train'][0][0]
    z_out_p = None
    results = {}
    
    for epoch in xrange(num_epochs):
        u = 0
        start_time = time.time()
        widgets = ['Epoch {}, '.format(epoch), Timer(), Bar()]
        pbar = ProgressBar(
            widgets=widgets, maxval=(train_samples // batch_size)).start()
        
        e_results = {}
        iterator = datasets['train'].get_epoch_iterator()
        
        for batch in iterator:
            x_in, label = batch
            
            if batch_size is None:
                batch_size = x_in.shape[0]
            
            if x_in.shape[0] != batch_size:
                break
                        
            if z_out_p is None:
                z_out_p = rng.normal(size=(batch_size, DIM_Z)).astype(floatX)

            if persist_p_chain:
                z_in_new = rng.normal(size=(batch_size, DIM_Z)).astype(floatX)
                blending = rng.uniform(0.0, 1.0, size=(batch_size,))
                z_in_new[
                    blending >= blending_rate] = z_out_p[
                    blending >= blending_rate]
                z_in = z_in_new
            else:
                z_in = rng.normal(size=(batch_size, DIM_Z)).astype(floatX)

            if latent_sparse:
                z_in[:, latent_sparse_num:] *= 0.0
            if MODULE._semi_supervised:
                outs = train_fn(x_in, z_in, label)
            else:
                outs = train_fn(x_in, z_in)
            update_dict_of_lists(e_results, **outs)
            
            u += 1
            pbar.update(u)
            
        # Results
        e_results = dict((k, np.mean(v)) for k, v in e_results.items())
        update_dict_of_lists(results, **e_results)
        
        print_section('Epoch {} completed')
        logger.info('Epoch {} of {} took {:.3f}s'.format(
            epoch + 1, num_epochs, time.time() - start_time))
        logger.info(e_results)
        
        # Save
        if archive:
            suffix = '_{}'.format(epoch)
        else:
            suffix = ''
        logger.debug('Saving to {}'.format(binary_dir))
        save_fn(out_path=binary_dir, suffix=suffix)
        
        # Images
        logger.debug('Saving images to {}'.format(image_dir))
        iterator = datasets['test'].get_epoch_iterator()
        x_in_, label_ = iterator.next()
        z_im = rng.normal(size=(x_in_.shape[0], DIM_Z)).astype(floatX)
        x_gen, y_gen = gen_fn(z_im)
        x_gen = R_NONLINEARITY(x_gen)
        x_chain = chain_fn(z_im)
        x_chain = x_chain[:(len(x_chain) // 2)]
        y_x_chain = inpaint_fn_x(np.array(np.eye(10).astype(floatX)), z_im[:10])
        
        x_y_chain = inpaint_fn_y(x_in_, z_im)
        x_y_chain = np.array(
            [np.argmax(x_y, axis=1) for x_y in x_y_chain]).transpose(1, 0)
        x_y_chain = x_y_chain[:64]
        #print x_y_chain
        xy_labels = []
        for c in x_y_chain:
            prob = np.bincount(c, minlength=DIM_L) / float(c.shape[0])
            args = np.argsort(prob)[::-1]
            s = np.sort(prob)[::-1]
            lab = ', '.join(['{}({:.2f})'.format(a_, s_)
                             for a_, s_ in zip(args, s)][:2])
            xy_labels.append(lab)
        #print xy_labels
        
        #x_chain_x = chain_fn_x(x_in_[:64])
        #inpaint_chain = inpaint_fn(x_in_[:64], z_im)
        #inpaint_chain_d = inpaint_fn_d(x_in_[:64], z_im)
        
        plot_images(
            x_gen[:64], path.join(image_dir, 'gen_epoch_{}'.format(epoch)),
            labels=np.argmax(y_gen[:64], axis=1))
        plot_images(R_NONLINEARITY(x_in_[:64]), path.join(image_dir, 'gt'),
                    labels=np.argmax(label_[:64], axis=1))
        plot_images(R_NONLINEARITY(x_in_[:64]), path.join(image_dir, 'gt_y'),
                    labels=xy_labels)
        
        #assert False
        '''
        chain = []
        for x in inpaint_chain:
            x = x.reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x_ = np.zeros((16, 8, DIM_C, DIM_X, DIM_Y))
            x_[:8] = x
            x_[8:] = x_in_[:64].reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x = x_.transpose(0, 3, 1, 4, 2)
            x = x.reshape(16 * DIM_X, 8 * DIM_Y, DIM_C)
            x = R_NONLINEARITY(x)
            chain.append(x)
        
        imageio.mimsave(path.join(image_dir, 'gen_inpaint.gif'), chain)
        
        chain = []
        for x in inpaint_chain_d:
            x = x.reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x_ = np.zeros((16, 8, DIM_C, DIM_X, DIM_Y))
            x_[:8] = x
            x_[8:] = x_in_[:64].reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x = x_.transpose(0, 3, 1, 4, 2)
            x = x.reshape(16 * DIM_X, 8 * DIM_Y, DIM_C)
            x = R_NONLINEARITY(x)
            chain.append(x)
        
        imageio.mimsave(path.join(image_dir, 'gen_inpaint_d.gif'), chain)
        
        '''
        chain = []
        for x in y_x_chain:
            x = x.reshape(2, 5, DIM_C, DIM_X, DIM_Y)
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape(2 * DIM_X, 5 * DIM_Y, DIM_C)
            x = R_NONLINEARITY(x)
            chain.append(x)
        
        imageio.mimsave(path.join(image_dir, 'gen_y_x_chain.gif'), chain)
        
        '''
        chain = []
        for x in x_chain_x:
            x = x.reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape(8 * DIM_X, 8 * DIM_Y, DIM_C)
            x = R_NONLINEARITY(x)
            chain.append(x)
            
        imageio.mimsave(path.join(image_dir, 'gen_chain_x.gif'), chain)
        '''

    
# MAIN -------------------------------------------------------------------------
    
_model_defaults = dict(
    num_steps=3,
    pd_steps=1,
    dim_z=128,
    loss='bgan',
    n_samples=10,
    start_on_x=False,
)

_optimizer_defaults = dict(
    optimizer='rmsprop',
    op_args=dict(learning_rate=1e-4)
)

_data_defaults = dict(
    batch_size=64,
    pad_to=None,#(32, 32),
    nonlinearity='tanh'
)

_train_defaults = dict(
    persist_p_chain=False,
    latent_sparse=False,
    blending_rate=0.5,
    latent_sparse_num=128,
    num_epochs=1000,
    archive=False
)

_visualize_defaults = dict(
    num_steps_long=40,
    visualize_every_update=0, # Not used yet
    noise_damping=0.9
)


def test(datasets, num_steps=None, pd_steps=None, loss=None,
         n_samples=None, start_on_x=None, **model_args):
    '''Test model and graph.
    
    '''
    
    if MODULE._semi_supervised and loss != 'bgan':
        raise NotImplementedError('Currently, semi-supervised is only '
                                  'implemented with BGAN (got {})'.format(
                                    loss))
    
    logger.info("Initializing parameters from {}".format(MODULE.__name__))
    gparams = MODULE.init_gparams({}, **model_args)
    dparams = MODULE.init_dparams({}, **model_args)

    logger.info("Setting input variables")
    x_in = T.tensor4('x_in')
    z_in = T.matrix('z_in')
    y_in = T.matrix('y_in')
    inputs = [x_in, z_in, y_in]
    
    d = MODULE.z_to_x(gparams, z_in, return_tensors=True, **model_args)
    d.update(MODULE.x_to_z(gparams, x_in, y_in, return_tensors=True,
                           **model_args))
    
    if MODULE._semi_supervised:
        x = d['x_f']
        p = d['y_f']
        z = d['z_f']
        p_e = T.tile(p[None, :, :], (n_samples, 1, 1))
        d['p_e'] = p_e
        p_r = p_e.reshape((-1, DIM_L))
        d['p_r'] = p_r
        x_e = T.tile(x[None, :, :, :, :], (n_samples, 1, 1, 1, 1))
        d['x_e'] = x_e
        z_e = T.tile(z[None, :, :], (n_samples, 1, 1))
        d['z_e'] = z_e
        x_r = x_e.reshape((-1, DIM_C, DIM_X, DIM_Y))
        d['x_r'] = x_r
        y_r = sample_multinomial(p_r)
        d['y_r'] = y_r
        y = y_r.reshape((n_samples, -1, DIM_L))
        d['y'] = y
        z_r = z_e.reshape((-1, DIM_Z))
        d['z_r'] = z_r
        
        d.update(**MODULE.discriminator(
            dparams, x_r, y_r, z_r,
            return_tensors=True, **model_args))
        
        dloss, gloss = bgan_loss_2(
            [d['d_ff_1']], [d['d_conv_1'].reshape((n_samples, x.shape[0], -1))], y, p)
        d['dloss'] = dloss
        d['gloss'] = gloss
    
    x, y = datasets['train'].get_epoch_iterator().next()
    z = rng.normal(size=(x.shape[0], DIM_Z)).astype(floatX)
    
    for k, v in d.items():
        print 'Trying {}'.format(k)
        f = theano.function(inputs, v, on_unused_input='ignore')
        print f(x, z, y).shape
    assert False, d
    

def reload_parameters(g_param_file=None, d_param_file=None):
    if g_param_file is not None:
        gparams = np.load(g_param_file)
    else:
        gparams = {}
        
    if dparam_file is not None:
        dparams = np.load(d_param_file)
        
    return gparams, dparams
    

def main(source, data_args, model_args, optimizer_args, train_args,
         visualize_args):

    datasets, data_shapes = prepare_data(args.source, **data_args)
    model_args.update(dim_x=DIM_X, dim_y=DIM_Y, dim_c=DIM_C)
    
    logger.info("Forming model with args {}".format(model_args))
    #test(datasets, **model_args)
    dloss, gloss, dparams, gparams, inputs, results, x_chain, y_chain = make_model(
        **model_args)
    
    updates = set_optimizer(dloss, gloss, dparams, gparams, **optimizer_args)
    
    train_fn = compile_train(updates, inputs, results)
    gen_fn = compile_generation(inputs[1], x_chain[-1], y_chain[-1])
    chain_fn = compile_chain(inputs[1], gparams,
                             num_steps_long=visualize_args['num_steps_long'],
                             **model_args)
    #chain_fn_x = compile_chain_x(inputs[0], gparams,
    #                             num_steps_long=visualize_args['num_steps_long'],
    #                             **model_args)
    chain_fn_x = None
    '''
    inpaint_fn = compile_inpaint_chain(
        inputs[0], inputs[1], gparams,
        num_steps_long=visualize_args['num_steps_long'], **model_args)
    
    inpaint_fn_d = compile_inpaint_chain(
        inputs[0], inputs[1], gparams,
        noise_damping=visualize_args['noise_damping'],
        num_steps_long=visualize_args['num_steps_long'], **model_args)
    '''
    
    inpaint_fn_x = compile_inpaint_x_chain(
        inputs[2], inputs[1], gparams,
        num_steps_long=visualize_args['num_steps_long'], **model_args)
    
    inpaint_fn_y = compile_inpaint_y_chain(
        inputs[0], inputs[1], gparams,
        num_steps_long=visualize_args['num_steps_long'], **model_args)
    
    inpaint_fn = None
    inpaint_fn_d = None
    
    def save(out_path=None, suffix=''):
        if out_path is None:
            return
        gparams_np = dict((k, v.eval()) for k, v in gparams.items())
        dparams_np = dict((k, v.eval()) for k, v in dparams.items())
        
        np.savez(path.join(out_path, 'g_params{}.npz'.format(suffix)),
                 **gparams_np)
        np.savez(path.join(out_path, 'd_params{}.npz'.format(suffix)),
                 **dparams_np)
    
    try:
        logger.info('Training with args {}'.format(train_args))
        train(train_fn, gen_fn,
              chain_fn, chain_fn_x,
              inpaint_fn, inpaint_fn_d, inpaint_fn_x, inpaint_fn_y,
              save, datasets, data_shapes, **train_args)
    except KeyboardInterrupt:
        logger.info('Training interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            

def config(data_args, model_args, optimizer_args, train_args, visualizer_args,
           config_file=None):
    if config_file is None:
        return
    
    raise NotImplementedError() # load yaml, update args with config dicts.

    
if __name__ == '__main__':
    MODULE = conv_ss_2
    parser = make_argument_parser()
    args = parser.parse_args()
    set_stream_logger(args.verbosity)
    out_paths = setup_out_dir(args.out_path, args.name)
    
    data_args = {}
    data_args.update(**_data_defaults)
    
    model_args = {}
    model_args.update(**MODULE._defaults)
    model_args.update(**_model_defaults)
    DIM_Z = model_args['dim_z']
    
    nonlinearity = data_args.pop('nonlinearity')
    if nonlinearity == 'sigmoid':
        model_args['nonlinearity'] = 'lambda x: tensor.nnet.sigmoid(x)'
        R_NONLINEARITY = lambda x: x
    elif nonlinearity == 'tanh':
        model_args['nonlinearity'] = 'lambda x: tensor.tanh(x)'
        R_NONLINEARITY = lambda x: 0.5 * (x + 1.)
        data_args['tanh'] = True
    else:
        raise
        
    optimizer_args = {}
    optimizer_args.update(**_optimizer_defaults)
    
    train_args = {}
    train_args.update(**_train_defaults)
    train_args['batch_size'] = data_args['batch_size']
    train_args.update(**out_paths)
    
    visualize_args = {}
    visualize_args.update(**_visualize_defaults)
    
    config(data_args, model_args, optimizer_args, train_args, visualize_args,
           config_file=args.config_file)
    
    main(args.source, data_args, model_args, optimizer_args, train_args,
         visualize_args)
    

