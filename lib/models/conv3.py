'''Convolutional model.

This model convolves / transpose convolves by effectively halving / doubling
the input. Latent variables z are Gaussian and each z is sampled from parameters
which are produced by x-z. In addition, noise is added to z-x through
concatenation. Finally, there is an option to use multiple discriminator scores.

'''

from collections import OrderedDict
import logging

from theano import tensor as T

from nn_layers import (
    fflayer, convlayer, param_init_fflayer, param_init_convlayer)
from utils import init_tparams, join2, srng


logger = logging.getLogger('UDGAN.conv3')

_semi_supervised = False
_defaults = dict(
    n_levels=3, dim_z=128, dim_h=128, dim_hd=512, multi_discriminator=True,
    normalize_z=False
)

DPARAMS = {}
GPARAMS = {}


# INITIALIZE PARAMETERS ########################################################

def init_gparams(n_levels=None, dim_z=None, dim_h=None, dim_c=None, dim_x=None,
                 dim_y=None, **kwargs):
    '''Initialize the generator parameters.
    
    '''
    logger.info('Initializing generator parameters.')
    global GPARAMS
    p = {}
    
    scale = 2 ** (n_levels)
    
    if not (dim_x / float(scale)).is_integer():
        logger.error(
            'X dim will not divide evenly with number of convolutional layers.')
        raise TypeError('X dim will not divide evenly with number of '
                        'convolutional layers. ({})'.format(
                            dim_x / float(scale)))
    
    if not (dim_y / float(scale)).is_integer():
        logger.error(
            'Y dim will not divide evenly with number of convolutional layers.')
        raise TypeError('Y dim will not divide evenly with number of '
                        'convolutional layers. ({})'.format(
                            dim_y / float(scale)))

    dim_in = dim_z * 2
    dim_out = dim_h * scale // 2
    p = param_init_fflayer(
        params=p, prefix='z_x_ff', nin=dim_in,
        nout=dim_out * dim_x // scale * dim_y // scale, ortho=False,
        batch_norm=True)

    dim_in = dim_out
    for level in xrange(n_levels):
        name = 'z_x_conv_{}'.format(level)
        if level == n_levels - 1:
            dim_out = dim_c
            batch_norm = False
        else:
            dim_out = dim_in // 2
            batch_norm = True
        
        p = param_init_convlayer(
            params=p, prefix=name, nin=dim_in, nout=dim_out,
            kernel_len=5, batch_norm=batch_norm)
        dim_in = dim_out

    for level in xrange(n_levels):
        name = 'x_z_conv_{}'.format(level)

        if level == 0:
            dim_out = dim_h
        else:
            dim_out = dim_in * 2
        
        p = param_init_convlayer(
            params=p, prefix=name, nin=dim_in, nout=dim_out,
            kernel_len=5, batch_norm=True)
        dim_in = dim_out

    p = param_init_fflayer(
        params=p, prefix='x_z_mu',
        nin=dim_h * dim_x * dim_y // scale // 2, nout=dim_z, ortho=False,
        batch_norm=False)

    p = param_init_fflayer(
        params=p, prefix='x_z_logsigma',
        nin=dim_h * dim_x * dim_y // scale // 2, nout=dim_z, ortho=False,
        batch_norm=False)

    GPARAMS = init_tparams(p)
    return GPARAMS


# MODELS #######################################################################

def z_to_x(z, n_levels=None, dim_h=None, dim_x=None, dim_y=None,
           return_tensors=False, **kwargs):
    '''z to x transition.
    
    '''
    p = GPARAMS
    scale = 2 ** n_levels
    d = OrderedDict()
    
    logger.debug("Added extra noise input")
    z = T.concatenate([z, srng.normal(size=z.shape)], axis=1)
    d['z_i'] = z
    x = fflayer(
        tparams=p, state_below=z, prefix='z_x_ff',
        activ='lambda x: tensor.nnet.relu(x, alpha=0.02)')
    d['x0_z'] = x

    x = x.reshape((-1, dim_h * scale // 2, dim_x // scale, dim_y // scale))
    d['x0_rs'] = x

    for level in xrange(n_levels):
        name = 'z_x_conv_{}'.format(level)
        logger.debug('Forming layer with name {}'.format(name))
        if level == n_levels - 1:
            activ = 'lambda x: T.tanh(x)'
        else:
            activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'

        x = convlayer(
            tparams=p, state_below=x, prefix=name,
            activ=activ, stride=-2)
        d['x{}'.format(level + 1)] = x

    if return_tensors:
        return d
    else:
        return x


def x_to_z(x, n_levels=None, dim_z=None, dim_h=None, dim_c=None, dim_x=None,
           dim_y=None, normalize_z=None, return_tensors=False, **kwargs):
    '''x to z transition.
    
    '''
    p = GPARAMS
    out = OrderedDict()
    h = x.reshape((-1, dim_c, dim_x, dim_y))

    for level in xrange(n_levels):
        name = 'x_z_conv_{}'.format(level)
        activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'
        h = convlayer(
            tparams=p, state_below=h, prefix=name, activ=activ,
            stride=2)
        out['h{}'.format(level)] = h

    h = h.flatten(2)

    log_sigma = fflayer(
        tparams=p, state_below=h, prefix='x_z_logsigma',
        active='T.tanh(x)')#activ='lambda x: x')
    out['logsigma'] = log_sigma

    mu = fflayer(
        tparams=p, state_below=h, prefix='x_z_mu',
        active='T.tanh(x)')#activ='lambda x: x')
    out['mu'] = mu

    eps = srng.normal(size=log_sigma.shape)
    z = eps * T.exp(log_sigma) + mu
    #z = eps * T.nnet.sigmoid(log_sigma) + mu # Old way
    out['z'] = z

    if normalize_z:
        logger.debug('Normalizing z')
        z = (z - T.mean(z, axis=0, keepdims=True)) / (
            1e-6 + T.std(z, axis=0, keepdims=True))
        out['z_norm'] = z

    if return_tensors:
        return out
    else:
        return z


def init_dparams(n_levels=None, dim_z=None, dim_h=None, dim_hd=None, dim_c=None,
                 dim_x=None, dim_y=None, multi_discriminator=None, **kwargs):
    '''Initialize the discriminator parameters.
    
    '''
    global DPARAMS
    p = {}

    logger.info('Initializing discriminator parameters.')
    if multi_discriminator:
        logger.info('Forming parameters for multiple scores.')
    
    dim_in = dim_c
    scale = 2 ** n_levels

    for level in xrange(n_levels):
        
        if level == 0:
            dim_out = dim_h
        else:
            dim_out = dim_in * 2
            
        batch_norm = (level != 0)

        p = param_init_convlayer(
            params=p, prefix='d_conv_{}'.format(level),
            nin=dim_in, nout=dim_out,
            kernel_len=5, batch_norm=batch_norm)

        dim_in = dim_out
        
    n_levels = 2
    dim_out = dim_hd
    for level in xrange(n_levels):
        
        if level == 0:
            dim_in = dim_h * scale * dim_x // scale * dim_y // scale // 2
        
        p = param_init_fflayer(
            params=p, prefix='d_ff_{}'.format(level), nin=dim_in, nout=dim_out,
            ortho=False, batch_norm=True)

        if level != n_levels - 1:
            p = param_init_fflayer(
                params=p, prefix='d_ff_z_{}'.format(level),
                nin=dim_z, nout=dim_out,
                ortho=False, batch_norm=True)
            
        if level == n_levels - 1 or multi_discriminator:
            name_out = 'd_ff_out_{}'.format(level)
            p = param_init_fflayer(
                params=p, prefix=name_out, nin=dim_out, nout=1,
                ortho=False, batch_norm=False)

        dim_in = dim_out
        dim_out = dim_out // 2
    
    DPARAMS = init_tparams(p)
    return DPARAMS


def discriminator(x, z, n_levels=None, dim_z=None, dim_h=None, dim_hd=None,
                  dim_c=None, dim_x=None, dim_y=None, multi_discriminator=None,
                  return_tensors=False, **kwargs):
    '''Discriminator function.
    
    '''
    p = DPARAMS
    outs = OrderedDict()
    
    h = x.reshape((-1, dim_c, dim_x, dim_y))
    outs['h0'] = h
    activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'

    ds = []
    for level in xrange(n_levels):
        h = convlayer(
            tparams=p, state_below=h, prefix='d_conv_{}'.format(level),
            activ=activ, stride=2)
        outs['h_conv_{}'.format(level)] = h
        
    h = h.flatten(2)
    outs['hf'] = h

    n_levels = 2
    for level in xrange(n_levels):
        h = fflayer(
            tparams=p, state_below=h, prefix='d_ff_{}'.format(level),
            activ=activ, mean_ln=False)
        outs['h_ff_{}'.format(level)] = h
        
        if level != n_levels - 1:
            z_h = fflayer(
                tparams=p, state_below=z, prefix='d_ff_z_{}'.format(level),
                activ=activ, mean_ln=False)
            h = z_h + h
        
        if level == n_levels - 1 or multi_discriminator:
            d = fflayer(
                tparams=p, state_below=h, prefix='d_ff_out_{}'.format(level),
                activ='lambda x: x')
            ds.append(d)
            outs['d_ff_{}'.format(level)] = d
            
    if return_tensors:
        return outs
    else:
        return ds, h
