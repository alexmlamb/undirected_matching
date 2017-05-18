#!/usr/bin/env python

'''Convolutional undirected GAN.

.. note::
   Initially make z_to_x and x_to_z fairly shallow networks.  Inject noise?fdfsj
   Use the fflayer class?

'''

<<<<<<< HEAD
=======
import theano
import theano.tensor as T
from nn_layers import fflayer, param_init_fflayer, param_init_convlayer, convlayer
from utils import init_tparams, join2, srng, dropout, inverse_sigmoid, join3, merge_images
from loss import accuracy, crossent, lsgan_loss, wgan_loss, improvement_loss
import lasagne
import numpy as np
import numpy.random as rng
>>>>>>> 21728d6c35a0c1e2c3ab22566046a38c999cc169
import gzip
import cPickle as pickle
import random
import os
from os import path
import sys
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

<<<<<<< HEAD
logger = logging.getLogger('UDGAN')
=======
import os
slurm_name = os.environ["SLURM_JOB_ID"]

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()

dataset = "mnist"
#dataset = "anime"
#dataset = "svhn"

if dataset == "mnist":
    mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

    train, valid, test = pickle.load(mn)

    trainx,trainy = train
    
    
    #newtx = trainx[(trainy<2) | (trainy>8)]
    #newty = trainy[(trainy<2) | (trainy>8)]
    #trainx = newtx
    #trainy = newty
    
    validx,validy = valid
    testx, testy = test

    num_examples = trainx.shape[0]

    m = 784
elif dataset == "anime":
    from load_file import FileData, normalize, denormalize

    loc = "/u/lambalex/DeepLearning/animefaces/datafaces/danbooru-faces/"

    animeData = FileData(loc, 32, 64)

    m = 32*32*3

elif dataset == "svhn":

    from load_svhn import SvhnData
    from load_file import normalize, denormalize

    svhnData = SvhnData(mb_size=64,segment="train")

    num_examples = 50000

nl = 128
print "num latent", nl
#128 works for nl
nfg = 512
nfd = 512

print "dataset", dataset

#3
num_steps = 1
print "num steps", num_steps

train_classifier_separate = True
print "train classifier separate", train_classifier_separate

#skip_conn = True
#print "skip conn", skip_conn

latent_sparse = False
print "latent sparse", latent_sparse

persist_p_chain = False
print "persistent p chain", persist_p_chain

blending_rate = 0.5
print 'blending rate (odds of keeping old z in P chain)', blending_rate

improvement_loss_weight = 0.0
print "improvement loss weight", improvement_loss_weight

def init_gparams(p):

    p = param_init_fflayer(options={},params=p,prefix='z_x_1',nin=nl*2,nout=512*4*4,ortho=False,batch_norm=True)

    p = param_init_convlayer(options={},params=p,prefix='z_x_2',nin=512,nout=256,kernel_len=5,batch_norm=True)
    p = param_init_convlayer(options={},params=p,prefix='z_x_3',nin=256*1,nout=128,kernel_len=5,batch_norm=True)
    p = param_init_convlayer(options={},params=p,prefix='z_x_4',nin=128*1,nout=3,kernel_len=5,batch_norm=False)

    p = param_init_convlayer(options={},params=p,prefix='x_z_1',nin=3,nout=128,kernel_len=5,batch_norm=True)
    p = param_init_convlayer(options={},params=p,prefix='x_z_2',nin=128,nout=256,kernel_len=5,batch_norm=True)
    p = param_init_convlayer(options={},params=p,prefix='x_z_3',nin=256,nout=512,kernel_len=5,batch_norm=True)

    p = param_init_fflayer(options={},params=p,prefix='x_z_mu',nin=512*4*4,nout=nl,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='x_z_sigma',nin=512*4*4,nout=nl,ortho=False,batch_norm=False)

    return init_tparams(p)

def init_dparams(p):

    p = param_init_convlayer(options={},params=p,prefix='DC_1',nin=3,nout=128,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='DC_2',nin=128,nout=256,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='DC_3',nin=256,nout=512,kernel_len=5,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='D_1',nin=nl+512*4*4,nout=nfd,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_2',nin=nfd,nout=nfd,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_3',nin=nfd,nout=nfd,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='D_o_1',nin=nfd,nout=1,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_2',nin=nfd,nout=1,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_3',nin=nfd,nout=1,ortho=False,batch_norm=False)

    p = param_init_convlayer(options={},params=p,prefix='D_o_4',nin=128,nout=1,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='D_o_5',nin=256,nout=1,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='D_o_6',nin=512,nout=1,kernel_len=5,batch_norm=False)

    return init_tparams(p)


def z_to_x(p,z):

    print "extra noise input"
    z_inp = join2(z, 1.0*srng.normal(size=z.shape))

    d0 = fflayer(tparams=p,state_below=z_inp,options={},prefix='z_x_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)')

    d0 = d0.reshape((64,512,4,4))

    d1 = convlayer(tparams=p,state_below=d0,options={},prefix='z_x_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=-2)

    d2 = convlayer(tparams=p,state_below=d1,options={},prefix='z_x_3',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=-2)

    d3 = convlayer(tparams=p,state_below=d2,options={},prefix='z_x_4',activ='lambda x: x',stride=-2)

    x_new = d3.flatten(2)

    return x_new

def x_to_z(p,x):

    e1 = convlayer(tparams=p,state_below=x.reshape((64,3,32,32)),options={},prefix='x_z_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=2)

    e2 = convlayer(tparams=p,state_below=e1,options={},prefix='x_z_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=2)

    e3 = convlayer(tparams=p,state_below=e2,options={},prefix='x_z_3',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=2)

    eo = e3
    eo = eo.flatten(2)

    sigma = fflayer(tparams=p,state_below=eo,options={},prefix='x_z_mu',activ='lambda x: x')
    mu = fflayer(tparams=p,state_below=eo,options={},prefix='x_z_sigma',activ='lambda x: x')

    eps = srng.normal(size=sigma.shape)

    z_new = eps*T.nnet.sigmoid(sigma) + mu
    print "turned on injected noise in x->z connection"

    z_new = (z_new - T.mean(z_new, axis=0, keepdims=True)) / (0.001 + T.std(z_new, axis=0, keepdims=True))

    return z_new


def discriminator(p,x,z):

    dc_1 = convlayer(tparams=p,state_below=x.reshape((64,3,32,32)),options={},prefix='DC_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=2)
>>>>>>> 21728d6c35a0c1e2c3ab22566046a38c999cc169

from data import load_stream, Pad
from exptools import make_argument_parser, setup_out_dir
from loggers import set_stream_logger
from loss import (accuracy, bgan_loss, bgan_loss_2, crossent, lsgan_loss,
                  wgan_loss)
from models import conv1, conv2
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


def compile_generation(z, x):
    logger.info("Compiling generation function")
    f = theano.function([z], x)
    return f


def compile_chain(z, gparams, num_steps_long=None, **model_args):
    logger.info("Compiling chain function")
    p_lst_x_long, p_lst_y_long, p_lst_z_long = p_chain(
        gparams, z, num_steps_long, **model_args)
    
    f = theano.function([z], outputs=p_lst_x_long + p_lst_y_long)
    
    return f


def compile_chain_x(x, gparams, num_steps_long=None, **model_args):
    logger.info("Compiling chain starting at x function")
    p_lst_x_long, p_lst_y_long, p_lst_z_long = p_chain(
        gparams, None, num_steps_long, x=x, **model_args)
    
    f = theano.function([x], outputs=p_lst_x_long + p_lst_y_long)
    
    return f


def compile_inpaint_chain(x, z, gparams, num_steps_long=None,
                          noise_damping=None, **model_args):
    logger.info("Compiling inpaint function")
    p_lst_x_long, p_lst_y_long = inpaint_chain(
        gparams, x, z, num_steps_long, noise_damping=noise_damping,
        **model_args)
    
    f = theano.function([x, z], outputs=p_lst_x_long + p_lst_y_long)
    
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

def p_chain(p, z, num_iterations, pd_steps=None, x=None, **model_args):
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
            x, p = out
            plst.append(p)
            y = sample_multinomial(y)
        else:
            x = out
            
        xlst.append(x)

        if i < num_iterations - 1:
            if MODULE._semi_supervised:
                if i == num_iterations - pd_steps - 1:
                    z = x_to_z(p, consider_constant(x), y, **model_args) # Changed this
                else:
                    z = x_to_z(p, x, y, **model_args)
            else:
                if i == num_iterations - pd_steps - 1:
                    z = x_to_z(p, consider_constant(x), **model_args) # Changed this
                else:
                    z = x_to_z(p, x, **model_args)
            
            zlst.append(z)

<<<<<<< HEAD
    #for j in range(len(xlst)):
    #    xlst[j] = T.nnet.sigmoid(xlst[j])
    
    assert len(xlst) == len(zlst)
    return xlst, plst, zlst
=======
    if num_iterations == 1:
        
        new_x = z_to_x(p, zlst[-1])
        xlst.append(new_x)
        #new_z = x_to_z(p, xlst[-1])
        #zlst.append(new_z)

    elif num_iterations == 3:  

        new_x = z_to_x(p, zlst[-1])
        xlst.append(new_x)
        new_z = x_to_z(p, consider_constant(xlst[-1]))
        zlst.append(new_z)

        new_x = z_to_x(p, zlst[-1])
        xlst.append(new_x)
        new_z = x_to_z(p, consider_constant(xlst[-1]))
        zlst.append(new_z)

        new_x = z_to_x(p, zlst[-1])
        xlst.append(new_x)

    else:

        for inds in range(0,num_iterations):
            new_x = z_to_x(p, zlst[-1])
            xlst.append(new_x)
            new_z = x_to_z(p, xlst[-1])
            zlst.append(new_z)


    for j in range(len(xlst)):
        xlst[j] = T.nnet.sigmoid(xlst[j])

    return xlst, zlst

def onestep_z_to_x(p,z):
    x = T.nnet.sigmoid(z_to_x(p, z))
    return x

def onestep_x_to_z(p,x):
    new_z = x_to_z(p, inverse_sigmoid(x))
    return new_z
>>>>>>> 21728d6c35a0c1e2c3ab22566046a38c999cc169


def q_chain(p, x, y, num_iterations, test=False, **model_args):
    x_to_z = MODULE.x_to_z
    xlst = [x]
    ylst = [y]
    zlst = []
    
    #x_ = inverse_sigmoid(x)
    x_ = x
    
    if MODULE._semi_supervised:
        new_z = x_to_z(p, x_, y)
    else:
        new_z = x_to_z(p, x_, **model_args)
        
    zlst.append(new_z)

    return xlst, ylst, zlst


<<<<<<< HEAD
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
            x, p = out
            y = sample_multinomial(p)
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
=======
print p_lst_x
print p_lst_z
print q_lst_x
print q_lst_z

#D_p_lst_3,_ = discriminator(dparams, p_lst_x[2], p_lst_z[2])

#D_p_lst_2,_ = discriminator(dparams, p_lst_x[1], p_lst_z[1])

D_p_lst_1,_ = discriminator(dparams, p_lst_x[0], p_lst_z[0])

D_q_lst,D_feat_q = discriminator(dparams, q_lst_x[-1], q_lst_z[-1])

dloss, gloss = lsgan_loss(D_q_lst, D_p_lst_1)

print "single disc"
print "not using improvement objective"
#improvement_objective = improvement_loss_weight * improvement_loss(D_p_lst_1, D_p_lst_2)
#gloss += improvement_objective

dupdates = lasagne.updates.rmsprop(dloss, dparams.values(),0.0001)
gloss_grads = T.grad(gloss, gparams.values(), disconnected_inputs='ignore')
gupdates = lasagne.updates.rmsprop(gloss_grads, gparams.values(),0.0001)

gcupdates = lasagne.updates.rmsprop(gloss, gparams.values(),0.0001)

dgupdates = dupdates.copy()
dgupdates.update(gupdates)

dgcupdates = dupdates.copy()
dgcupdates.update(gcupdates)

train_disc_gen_classifier = theano.function(inputs = [x_in, z_in], outputs=[dloss,p_lst_x[-1],p_lst_z[-1]], updates=dgcupdates,on_unused_input='ignore')

get_zinf = theano.function([x_in], outputs=z_inf)
#get_dfeat = theano.function([x_in], outputs=D_feat_q)

#get_pchain = theano.function([z_in], outputs = p_lst_x_long)

x_in = T.matrix()

func_z_to_x = theano.function([z_in], outputs = onestep_z_to_x(gparams, z_in))
func_x_to_z = theano.function([x_in], outputs = onestep_x_to_z(gparams, x_in))

if __name__ == '__main__':

    z_out_p = rng.normal(size=(64,nl)).astype('float32')

    for iteration in range(0,500000):

        if persist_p_chain:
            z_in_new = rng.normal(size=(64,nl)).astype('float32')
            blending = rng.uniform(0.0,1.0,size=(64,))
            z_in_new[blending>=blending_rate] = z_out_p[blending>=blending_rate]
            z_in = z_in_new
        else:
            z_in = rng.normal(size=(64,nl)).astype('float32')

        if latent_sparse:
            z_in[:,128:] *= 0.0

        r = random.randint(0,num_examples-64)
        
        if dataset == "mnist":
            x_in = trainx[r:r+64]
            y_in = trainy[r:r+64]
>>>>>>> 21728d6c35a0c1e2c3ab22566046a38c999cc169

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
        x, p = z_to_x(p, z, **model_args)
        p = sample_multinomial(y)
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
        x, p = z_to_x(p, z, **model_args)
        xlst.append(x)
        z = x_to_z(p, x, y_gt, **model_args)
        
    return xlst


<<<<<<< HEAD
def onestep_z_to_x(p, z, **model_args):
    out = MODULE.z_to_x(p, z, **model_args)
    if MODULE._semi_supervised:
        x, y = out
    else:
        x = out
    #x = T.nnet.sigmoid(x)
    return x
=======
        dloss,gen_x,z_out_p = train_disc_gen_classifier(x_in,z_in)
        
>>>>>>> 21728d6c35a0c1e2c3ab22566046a38c999cc169


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
        p_lst_x, p_lst_p, p_lst_z = p_chain(gparams, z_in, num_steps,
                                        pd_steps=pd_steps, **model_args)
        p_lst_x_ = p_lst_x

    # p_list_y_ is pre-activation and sampling
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
        p_lst_y = [y.reshape((n_samples, -1, DIM_L)) for y in l_lst_y_r]
        p_lst_z_r = [z.reshape((-1, DIM_Z))
                     for z in p_lst_z_e]
        
        for i in xrange(pd_steps):
            D_p_lst_, D_feat_p = MODULE.discriminator(
                dparams, p_lst_x_r[-(i + 1)], p_lst_y_r[-(i + 1)],
                p_lst_z_r[-(i + 1)], **model_args)
            D_p_lst += D_p_lst_
        D_p_lst = [D.reshape((n_samples, -1)) for D in D_p_lst]
        D_q_lst, D_feat_q = MODULE.discriminator(
            dparams, q_lst_x[-1], q_lst_y[-1], q_lst_z[-1], **model_args)
        
        d_loss = bgan_loss_2(D_p_lst, D_q_lst, p_lst_y, p_lst_p)
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

    return dloss, gloss, dparams, gparams, [x_in, z_in], results, p_lst_x_

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

def train(train_fn, gen_fn, chain_fn, chain_fn_x, inpaint_fn, inpaint_fn_d,
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

            outs = train_fn(x_in, z_in)
            update_dict_of_lists(e_results, **outs)
            
<<<<<<< HEAD
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
        iterator = datasets['test'].get_epoch_iterator()
        x_in_, label_ = iterator.next()
        logger.debug('Saving images to {}'.format(image_dir))
        z_im = rng.normal(size=(x_in_.shape[0], DIM_Z)).astype(floatX)
        x_gen = gen_fn(z_im)
        x_gen = 0.5 * (x_gen + 1.)
        x_chain = chain_fn(z_im)
        x_chain_x = chain_fn_x(x_in_)
        inpaint_chain = inpaint_fn(x_in_, z_im)
        inpaint_chain_d = inpaint_fn_d(x_in_, z_im)
        
        plot_images(
            x_gen[:64], path.join(image_dir, 'gen_epoch_{}'.format(epoch)))
        plot_images(x_in_[:64], path.join(image_dir, 'gt'))
        
        chain = []
        for x in inpaint_chain:
            x = x[:64]
            x = x.reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x_ = np.zeros((16, 8, DIM_C, DIM_X, DIM_Y))
            x_[:8] = x
            x_[8:] = x_in_[:64].reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x = x_.transpose(0, 3, 1, 4, 2)
            x = x.reshape(16 * DIM_X, 8 * DIM_Y, DIM_C)
            x = 0.5 * (x + 1.)
            chain.append(x)
        
        imageio.mimsave(path.join(image_dir, 'gen_inpaint.gif'), chain)
        
        chain = []
        for x in inpaint_chain_d:
            x = x[:64]
            x = x.reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x_ = np.zeros((16, 8, DIM_C, DIM_X, DIM_Y))
            x_[:8] = x
            x_[8:] = x_in_[:64].reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x = x_.transpose(0, 3, 1, 4, 2)
            x = x.reshape(16 * DIM_X, 8 * DIM_Y, DIM_C)
            x = 0.5 * (x + 1.)
            chain.append(x)
        
        imageio.mimsave(path.join(image_dir, 'gen_inpaint_d.gif'), chain)
        
        chain = []
        for x in x_chain:
            x = x[:64]
            x = x.reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape(8 * DIM_X, 8 * DIM_Y, DIM_C)
            x = 0.5 * (x + 1.)
            chain.append(x)
        
        imageio.mimsave(path.join(image_dir, 'gen_chain.gif'), chain)
        
        chain = []
        for x in x_chain_x:
            x = x[:64]
            x = x.reshape(8, 8, DIM_C, DIM_X, DIM_Y)
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape(8 * DIM_X, 8 * DIM_Y, DIM_C)
            x = 0.5 * (x + 1.)
            chain.append(x)
            
        imageio.mimsave(path.join(image_dir, 'gen_chain_x.gif'), chain)

    
# MAIN -------------------------------------------------------------------------
    
_model_defaults = dict(
    num_steps=3,
    pd_steps=1,
    dim_z=128,
    loss='bgan',
    n_samples=10,
    start_on_x=False
)

_optimizer_defaults = dict(
    optimizer='rmsprop',
    op_args=dict(learning_rate=1e-4)
)

_data_defaults = dict(
    batch_size=64,
    pad_to=None#(32, 32)
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
    chain_fn_x = compile_chain_x(inputs[0], gparams,
                                 num_steps_long=visualize_args['num_steps_long'],
                                 **model_args)
    
    inpaint_fn = compile_inpaint_chain(
        inputs[0], inputs[1], gparams,
        num_steps_long=visualize_args['num_steps_long'], **model_args)
    
    inpaint_fn_d = compile_inpaint_chain(
        inputs[0], inputs[1], gparams,
        noise_damping=visualize_args['noise_damping'],
        num_steps_long=visualize_args['num_steps_long'], **model_args)
    
    def save(out_path=None, suffix=''):
        if out_path is None:
            return
        
        np.savez(path.join(out_path, 'g_params{}.npz'.format(suffix)), gparams)
        np.savez(path.join(out_path, 'd_params{}.npz'.format(suffix)), dparams)
    
    try:
        logger.info('Training with args {}'.format(train_args))
        train(train_fn, gen_fn, chain_fn, chain_fn_x, inpaint_fn, inpaint_fn_d,
              save, datasets, data_shapes, **train_args)
    except KeyboardInterrupt:
        logger.info('Training interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
            
=======
            #p_chain = get_pchain(z_in)
            new_z = rng.normal(size=(64,nl)).astype('float32')
            for j in range(0,20):
                new_x = func_z_to_x(new_z)
                new_z = func_x_to_z(new_x)
                print "printing element of p_chain", j
                plot_images(new_x, "plots/" + slurm_name + "_pchain_" + str(j) + ".png")
        
            new_z = rng.normal(size=(64,nl)).astype('float32')
            for j in range(0,20):
                new_x = func_z_to_x(new_z)
                new_x = merge_images(new_x, x_in)
                new_z = func_x_to_z(new_x)
                plot_images(new_x, "plots/" + slurm_name + "_inpainting_" + str(j) + ".png")

>>>>>>> 21728d6c35a0c1e2c3ab22566046a38c999cc169

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
    DIM_Z = model_args['dim_z']
    
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
    

