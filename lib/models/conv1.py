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


logger = logging.getLogger('UDGAN.conv1')

_defaults = dict(
    n_levels=3, dim_z=128, dim_h=128, dim_hd=512, multi_discriminator=True,
    normalize_z=True
)

# INITIALIZE PARAMETERS ########################################################

def init_gparams(p, n_levels=3, dim_z=128, dim_h=128, dim_c=3, dim_x=32,
                 dim_y=32, **kwargs):
    '''Initialize the generator parameters.
    
    '''
    logger.info('Initializing generator parameters.')
    
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
    
    logger.debug('Generator z-x dim in / out: {}-{}'.format(
            dim_in, dim_out * dim_x // scale * dim_y // scale))
    p = param_init_fflayer(
        options={}, params=p, prefix='z_x_ff', nin=dim_in,
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

        logger.debug('Generator z-x dim in / out: {}-{}'.format(
            dim_in, dim_out))
        
        p = param_init_convlayer(
            options={}, params=p, prefix=name, nin=dim_in, nout=dim_out,
            kernel_len=5, batch_norm=batch_norm)
        dim_in = dim_out

    for level in xrange(n_levels):
        name = 'x_z_conv_{}'.format(level)

        if level == 0:
            dim_out = dim_h
        else:
            dim_out = dim_in * 2

        logger.debug('Generator x-z dim in / out: {}-{}'.format(
            dim_in, dim_out))
        
        p = param_init_convlayer(
            options={}, params=p, prefix=name, nin=dim_in, nout=dim_out,
            kernel_len=5, batch_norm=True)
        dim_in = dim_out

    p = param_init_fflayer(
        options={}, params=p, prefix='x_z_mu',
        nin=dim_h * dim_x * dim_y // scale // 2, nout=dim_z, ortho=False,
        batch_norm=False)

    p = param_init_fflayer(
        options={}, params=p, prefix='x_z_logsigma',
        nin=dim_h * dim_x * dim_y // scale // 2, nout=dim_z, ortho=False,
        batch_norm=False)

    return init_tparams(p)


def init_dparams(p, n_levels=3, dim_z=128, dim_h=128, dim_hd=512, dim_c=3,
                 dim_x=32, dim_y=32, multi_discriminator=True, **kwargs):
    '''Initialize the discriminator parameters.
    
    '''

    logger.info('Initializing discriminator parameters.')
    if multi_discriminator:
        logger.info('Forming parameters for multiple scores.')
    
    dim_in = dim_c
    scale = 2 ** n_levels

    for level in xrange(n_levels):
        name = 'd_conv_{}'.format(level)
        
        if level == 0:
            dim_out = dim_h
        else:
            dim_out = dim_in * 2
        
        logger.debug('Discriminator conv dim in / out: {}-{}'.format(
            dim_in, dim_out))

        p = param_init_convlayer(
            options={}, params=p, prefix=name, nin=dim_in, nout=dim_out,
            kernel_len=5, batch_norm=False)

        if multi_discriminator:
            name_out = 'd_conv_out_{}'.format(level)
            p = param_init_convlayer(
                options={}, params=p, prefix=name_out, nin=dim_out, nout=1,
                kernel_len=5, batch_norm=False)

        dim_in = dim_out

    for level in xrange(n_levels):
        name = 'd_ff_{}'.format(level)

        if level == 0:
            dim_in = dim_z + dim_h * scale * dim_x // scale * dim_y // scale // 2
        
        if level == n_levels - 1:
            dim_out = 1
        else:
            dim_out = dim_hd
            
        logger.debug('Discriminator ff dim in / out: {}-{}'.format(
            dim_in, dim_out))
        
        p = param_init_fflayer(
            options={}, params=p, prefix=name, nin=dim_in, nout=dim_out,
            ortho=False, batch_norm=False)

        if multi_discriminator or level == n_levels - 1:
            name_out = 'd_ff_out_{}'.format(level)
            p = param_init_fflayer(
                options={}, params=p, prefix=name_out, nin=dim_out, nout=1,
                ortho=False, batch_norm=False)

        dim_in = dim_out
    
    return init_tparams(p)


# MODELS #######################################################################

def z_to_x(p, z, n_levels=3, dim_h=128, dim_x=32, dim_y=32,
           return_tensors=False, **kwargs):
    '''z to x transition.
    
    '''
    scale = 2 ** n_levels
    d = OrderedDict()
    
    logger.debug("Added extra noise input")
    z = T.concatenate([z, srng.normal(size=z.shape)], axis=1)
    d['z'] = z
    logger.debug('Forming layer with name {}'.format('z_x_ff'))
    x = fflayer(
        tparams=p, state_below=z, options={}, prefix='z_x_ff',
        activ='lambda x: tensor.nnet.relu(x, alpha=0.02)')
    d['x0'] = x

    x = x.reshape((-1, dim_h * scale // 2, dim_x // scale, dim_y // scale))
    d['x0_rs'] = x

    for level in xrange(n_levels):
        name = 'z_x_conv_{}'.format(level)
        logger.debug('Forming layer with name {}'.format(name))
        if level == n_levels - 1:
            activ = 'lambda x: x'
        else:
            activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'

        x = convlayer(
            tparams=p, state_below=x, options={}, prefix=name,
            activ=activ, stride=-2)
        d['x{}'.format(level + 1)] = x

    if return_tensors:
        return d
    else:
        return x


def x_to_z(p, x, n_levels=3, dim_z=128, dim_h=128, dim_c=3, dim_x=32, dim_y=32,
           normalize_z=False, return_tensors=False, **kwargs):
    '''x to z transition.
    
    '''
    out = OrderedDict()
    h = x.reshape((-1, dim_c, dim_x, dim_y))

    for level in xrange(n_levels):
        name = 'x_z_conv_{}'.format(level)
        logger.debug('Forming layer with name {}'.format(name))
        activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'
        h = convlayer(
            tparams=p, state_below=h, options={}, prefix=name, activ=activ,
            stride=2)
        out['h{}'.format(level)] = h

    h = h.flatten(2)

    logger.debug('Forming layer with name {}'.format('x_z_logsigma'))
    log_sigma = fflayer(
        tparams=p, state_below=h, options={}, prefix='x_z_logsigma',
        activ='lambda x: x')
    out['logsigma'] = log_sigma

    logger.debug('Forming layer with name {}'.format('x_z_mu'))
    mu = fflayer(
        tparams=p, state_below=h, options={}, prefix='x_z_mu',
        activ='lambda x: x')
    out['mu'] = mu
    
    #log_sigma = T.minimum(log_sigma, 5.)

    eps = srng.normal(size=log_sigma.shape)
    #z = eps * T.exp(log_sigma) + mu
    z = eps * T.nnet.sigmoid(log_sigma) + mu # Old way
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


def discriminator(p, x, z, n_levels=3, dim_z=128, dim_h=128, dim_hd=512,
                  dim_c=3, dim_x=32, dim_y=32, multi_discriminator=True,
                  return_tensors=False, **kwargs):
    '''Discriminator function.
    
    '''
    if multi_discriminator:
        logger.info('Using mutliple scores for the discriminator.')
    else:
        logger.info('Using single score.')
    outs = OrderedDict()
    
    h = x.reshape((-1, dim_c, dim_x, dim_y))
    outs['h0'] = h
    activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'

    ds = []
    for level in xrange(n_levels):
        name = 'd_conv_{}'.format(level)
        logger.debug('Forming layer with name {}'.format(name))
        h = convlayer(
            tparams=p, state_below=h, options={}, prefix=name, activ=activ,
            stride=2)
        outs['h_conv_{}'.format(level)] = h
        
        if multi_discriminator:
            name_out = 'd_conv_out_{}'.format(level)
            d = convlayer(
                tparams=p, state_below=h, options={}, prefix=name_out,
                activ='lambda x: x', stride=2)
            ds.append(d)
            outs['d_conv_{}'.format(level)] = d
    outs['hf'] = h.flatten(2)
    outs['z'] = z
    h = T.concatenate([z, h.flatten(2)], axis=1)
    outs['h1'] = h

    for level in xrange(n_levels):
        name = 'd_ff_{}'.format(level)
        logger.debug('Forming layer with name {}'.format(name))
        h = fflayer(
            tparams=p, state_below=h, options={}, prefix=name, activ=activ,
            mean_ln=False)
        outs['h_ff_{}'.format(level)] = h
        if multi_discriminator or level == n_levels - 1:
            name_out = 'd_ff_out_{}'.format(level)
            d = fflayer(
                tparams=p, state_below=h, options={}, prefix=name_out,
                activ='lambda x: x')
            ds.append(d)
            outs['d_ff_{}'.format(level)] = d
    if return_tensors:
        return outs
    else:
        return ds, h
