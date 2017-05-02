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
import time

import lasagne
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
from viz import plot_images

logger = logging.getLogger('UDGAN')

from data import load_stream, Pad
from exptools import make_argument_parser, setup_out_dir
import imageio
from loggers import set_stream_logger
from loss import accuracy, crossent, lsgan_loss, wgan_loss
from models import conv1
from nn_layers import fflayer, convlayer
from progressbar import Bar, ProgressBar, Percentage, Timer
from utils import (
    init_tparams, join2, srng, dropout, inverse_sigmoid, join3, merge_images)


floatX = theano.config.floatX

try:
    SLURM_NAME = os.environ["SLURM_JOB_ID"]
except:
    SLURM_NAME = 'NA'
    
DIM_X = None
DIM_Y = None
DIM_C = None
DIM_Z = None
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
        #gloss_grads = T.grad(gloss, gparams.values(),
        #                     disconnected_inputs='ignore')
        # I don't think ignoring disconnected is a good idea.
        # Actually, I don't even know what this is for
        #gupdates = lasagne.updates.rmsprop(gloss_grads, gparams.values(),
        #                                   **op_args)
        gcupdates = lasagne.updates.rmsprop(gloss, gparams.values(), **op_args)
    else:
        raise NotImplementedError(optimizer)

    #dgupdates = dupdates.copy()
    #dgupdates.update(gupdates)

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


def compile_generation(z, x):
    logger.info("Compiling generation function")
    f = theano.function([z], x)
    return f


def compile_chain(z, gparams, num_steps_long=None, **model_args):
    logger.info("Compiling chain function")
    p_lst_x_long, p_lst_z_long = p_chain(
        gparams, z, num_steps_long, **model_args)
    
    f = theano.function([z], outputs=p_lst_x_long)
    
    return f


def compile_inpaint_chain(x, z, gparams, num_steps_long=None, **model_args):
    logger.info("Compiling inpaint function")
    p_lst_x_long = inpaint_chain(
        gparams, x, z, num_steps_long, **model_args)
    
    f = theano.function([x, z], outputs=p_lst_x_long)
    
    return f


def compile_piecewise_chain(z, x, gparams, **model_args):
    logger.info("Compiling piecewise chain function")
    f_z_to_x = theano.function([z], outputs=onestep_z_to_x(gparams, z))
    f_x_to_z = theano.function([x], outputs=onestep_x_to_z(gparams, x))
    
    return f_z_to_x, f_x_to_z
    
# VIZ --------------------------------------------------------------------------
    
def visualizer(num_steps_long=None):
    '''For eval, not for training.
    
    '''
    
    for j in range(0, len(p_chain)):
        print "printing element of p_chain", j
        plot_images(
            p_chain[j], "plots/" + slurm_name + "_pchain_" + str(j) +
            ".png")

    new_z = rng.normal(size=(64, nl)).astype('float32')
    for j in range(0, len(p_chain)):
        new_x = func_z_to_x(new_z)
        new_x = merge_images(new_x, x_in)
        new_z = func_x_to_z(new_x)
        plot_images(
            new_x,
            "plots/" + slurm_name +
            "_inpainting_" + str(j) + ".png") 

# MODELS -----------------------------------------------------------------------

def p_chain(p, z, num_iterations, pd_steps=None, **model_args):
    z_to_x = MODULE.z_to_x
    x_to_z = MODULE.x_to_z
    zlst = [z]
    xlst = []

    for i in xrange(num_iterations):
        x = z_to_x(p, z, **model_args)
        xlst.append(x)

        if i < num_iterations - 1:
            if i == num_iterations - pd_steps - 1:
                z = x_to_z(p, consider_constant(x), **model_args) # Changed this
            else:
                z = x_to_z(p, x, **model_args)
            
            zlst.append(z)

    for j in range(len(xlst)):
        xlst[j] = T.nnet.sigmoid(xlst[j])
    
    assert len(xlst) == len(zlst)
    return xlst, zlst


def inpaint_chain(p, x, z, num_iterations, **model_args):
    z_to_x = MODULE.z_to_x
    x_to_z = MODULE.x_to_z
    x_gt = inverse_sigmoid(x)
    xlst = [x_gt]
    
    for i in xrange(num_iterations):
        x = z_to_x(p, z, **model_args)
        x = T.set_subtensor(x[:, :, :DIM_X // 2, :], x_gt[:, :, :DIM_X // 2, :])
        xlst.append(x)
        z = x_to_z(p, x, **model_args)

    for j in range(len(xlst)):
        xlst[j] = T.nnet.sigmoid(xlst[j])
        
    return xlst


def onestep_z_to_x(p, z, **model_args):
    x = T.nnet.sigmoid(MODULE.z_to_x(p, z, **model_args))
    return x


def onestep_x_to_z(p, x, **model_args):
    new_z = x_to_z(p, inverse_sigmoid(x), **model_args)
    return new_z


def q_chain(p, x, num_iterations, test=False, **model_args):
    x_to_z = MODULE.x_to_z
    xlst = [x]
    zlst = []
    new_z = x_to_z(p, inverse_sigmoid(x), **model_args)
    zlst.append(new_z)

    return xlst, zlst


def make_model(num_steps=None, pd_steps=None, **model_args):
    '''Form the model and graph.
    
    '''
    
    logger.info("Initializing parameters from {}".format(MODULE.__name__))
    gparams = MODULE.init_gparams({}, **model_args)
    dparams = MODULE.init_dparams({}, **model_args)

    logger.info("Setting input variables")
    z_in = T.matrix('z_in')
    x_in = T.tensor4('x_in')

    logger.info("Building graph")
    
    p_lst_x, p_lst_z = p_chain(gparams, z_in, num_steps, pd_steps=pd_steps,
                               **model_args)
    q_lst_x, q_lst_z = q_chain(gparams, x_in, num_steps, **model_args)

    z_inf = q_lst_z[-1]

    logger.debug("p chain x: {}".format(p_lst_x))
    logger.debug("p chain z: {}".format(p_lst_z))
    logger.debug("q chain x: {}".format(q_lst_x))
    logger.debug("q chain z: {}".format(q_lst_z))

    logger.info('Using {} steps of p in discriminator'.format(pd_steps))
    D_p_lst = []
    for i in xrange(pd_steps):
        D_p_lst_, D_feat_p = MODULE.discriminator(
            dparams, p_lst_x[-(i + 1)], p_lst_z[-(i + 1)], **model_args)
        D_p_lst += D_p_lst_
    D_q_lst, D_feat_q = MODULE.discriminator(
        dparams, q_lst_x[-1], q_lst_z[-1], **model_args)
    
    dloss, gloss = lsgan_loss(D_q_lst, D_p_lst)
    
    results = {
        'D loss': dloss,
        'G loss': gloss,
        'p(real)': (q_lst_x[-1] > 0.5).mean(),
        'p(fake)': (p_lst_x[-1] < 0.5).mean()
    }

    return dloss, gloss, dparams, gparams, [x_in, z_in], results, p_lst_x

# DATA -------------------------------------------------------------------------

def prepare_data(source, pad_to=None, batch_size=None, **kwargs):
    logger.info('Perparing data from `{}`'.format(source))
    global DIM_X, DIM_Y, DIM_C
    
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

    shape = data_shapes['train'][0]
    
    logger.info('Setting DIM_X to {}, DIM_Y to {}, and DIM_C to {}'.format(
        shape[2], shape[3], shape[1]))
    DIM_X = shape[2]
    DIM_Y = shape[3]
    DIM_C = shape[1]
    
    return datasets, data_shapes

# TRAIN ------------------------------------------------------------------------

def train(train_fn, gen_fn, chain_fn, inpaint_fn, save_fn, datasets,
          data_shapes, persist_p_chain=None, latent_sparse=None,
          num_epochs=None, blending_rate=None, latent_sparse_num=None,
          batch_size=None, binary_dir=None, image_dir=None, archive=None):
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
        x_in_ = None
        
        for batch in iterator:
            x_in, label = batch
            if x_in_ is None:
                x_in_ = x_in
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
        z_im = rng.normal(size=(64, DIM_Z)).astype(floatX)
        x_gen = gen_fn(z_im)
        x_chain = chain_fn(z_im)
        inpaint_chain = inpaint_fn(x_in_[:64], z_im)
        
        plot_images(
            x_gen, path.join(image_dir, 'gen_epoch_{}.png'.format(epoch)))
        plot_images(x_in_[:64], path.join(image_dir, 'gt.png'))
        
        chain = []
        for x in inpaint_chain:
            x = x.reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape(8 * DIM_X, 8 * DIM_Y, DIM_C)
            chain.append(x)
        
        imageio.mimsave(path.join(image_dir, 'gen_inpaint.gif'), chain)
        
        chain = []
        for x in x_chain:
            x = x.reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape(8 * DIM_X, 8 * DIM_Y, DIM_C)
            chain.append(x)
        
        imageio.mimsave(path.join(image_dir, 'gen_chain.gif'), chain)
            
    
# MAIN -------------------------------------------------------------------------
    
_model_defaults = dict(
    num_steps=3,
    pd_steps=1,
    dim_z=128
)

_optimizer_defaults = dict(
    optimizer='rmsprop',
    op_args=dict(learning_rate=1e-4)
)

_data_defaults = dict(
    batch_size=64,
    pad_to=(32, 32)
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
    num_steps_long=20,
    visualize_every_update=0, # Not used yet
)


def test():
    '''
    TODO
    '''
    
    inputs, d = make_model(**model_args)
    x = datasets['train'].get_epoch_iterator().next()[0]
    z = rng.normal(size=(x.shape[0], DIM_Z)).astype(floatX)
    
    for k, v in d.items():
        print 'Trying {}'.format(k)
        f = theano.function(inputs, v, on_unused_input='ignore')
        print f(x, z).shape
    assert False, d
    

def main(source, data_args, model_args, optimizer_args, train_args,
         visualize_args):
    global DIM_Z
    DIM_Z = model_args['dim_z']
    
    datasets, data_shapes = prepare_data(args.source, **data_args)
    model_args.update(dim_x=DIM_X, dim_y=DIM_Y, dim_c=DIM_C)
    
    logger.info("Forming model with args {}".format(model_args))
    
    dloss, gloss, dparams, gparams, inputs, results, x_chain = make_model(
        **model_args)
    updates = set_optimizer(dloss, gloss, dparams, gparams, **optimizer_args)
    
    train_fn = compile_train(updates, inputs, results)
    gen_fn = compile_generation(inputs[1], x_chain[-1])
    chain_fn = compile_chain(inputs[1], gparams,
                             num_steps_long=visualize_args['num_steps_long'],
                             **model_args)
    
    inpaint_fn = compile_inpaint_chain(
        inputs[0], inputs[1], gparams,
        num_steps_long=visualize_args['num_steps_long'], **model_args)
    
    def save(out_path=None, suffix=''):
        if out_path is None:
            return
        
        np.savez(path.join(out_path, 'g_params{}.npz'.format(suffix)), gparams)
        np.savez(path.join(out_path, 'd_params{}.npz'.format(suffix)), dparams)
    
    try:
        logger.info('Training with args {}'.format(train_args))
        train(train_fn, gen_fn, chain_fn, inpaint_fn, save, datasets,
              data_shapes, **train_args)
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
    MODULE = conv1
    parser = make_argument_parser()
    args = parser.parse_args()
    set_stream_logger(args.verbosity)
    out_paths = setup_out_dir(args.out_path, args.name)
    
    data_args = {}
    data_args.update(**_data_defaults)
    
    model_args = {}
    model_args.update(**MODULE._defaults)
    model_args.update(**_model_defaults)
    
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
    

