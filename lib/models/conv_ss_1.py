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

_semi_supervised = True
_defaults = dict(
    n_levels=2, dim_z=128, dim_h=128, dim_hd=512, multi_discriminator=True,
    normalize_z=True, nonlinearity='lambda x: tensor.tanh(x)'
)

# INITIALIZE PARAMETERS ########################################################

def init_gparams(p, n_levels=3, dim_z=128, dim_h=128, dim_c=3, dim_x=32,
                 dim_y=32, dim_l=10, **kwargs):
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
    
    # Z-X Pathway
    # Transpose conv networks
    
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
        
    # X-Z Pathway
    # conv networks
        
    p = param_init_fflayer(
        options={}, params=p, prefix='n_z_ff_x', nin=dim_z,
        nout=dim_h * dim_x * dim_y // 4, ortho=False, batch_norm=True)
        
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
        
    # Z-Y Pathway
    # ff networks
        
    dim_in = dim_z * 2
    for level in xrange(2):
        name = 'z_y_ff_{}'.format(level)
        if level == 1:
            dim_out = dim_l
            batch_norm = False
        else:
            dim_out = dim_in // 2
            batch_norm = True
            
        logger.debug('Generator z-y dim in / out: {}-{}'.format(
            dim_in, dim_out))
        
        p = param_init_fflayer(
            options={}, params=p, prefix=name, nin=dim_in, nout=dim_out,
            batch_norm=batch_norm)
        dim_in = dim_out
        
    # Y-Z Pathway
    # ff networks
        
    p = param_init_fflayer(
        options={}, params=p, prefix='n_z_ff_y', nin=dim_z,
        nout=dim_h // 2, ortho=False, batch_norm=True)
        
    for level in xrange(2):
        name = 'y_z_ff_{}'.format(level)

        if level == 0:
            dim_out = dim_h // scale * 2
        else:
            dim_out = dim_in * 2
        logger.debug('Generator y-z dim in / out: {}-{}'.format(
            dim_in, dim_out))
        
        p = param_init_fflayer(
            options={}, params=p, prefix=name, nin=dim_in, nout=dim_out,
            batch_norm=True)
        dim_in = dim_out
        
    # Final Z Pathway
    # ff networks

    p = param_init_fflayer(
        options={}, params=p, prefix='x_z_final1',
        nin=dim_h * dim_x * dim_y // scale // 2, nout=dim_h, ortho=False,
        batch_norm=True)

    p = param_init_fflayer(
        options={}, params=p, prefix='x_z_final2',
        nin=2*dim_h, nout=dim_z, ortho=False,
        batch_norm=False)

    return init_tparams(p)


def init_dparams(p, n_levels=3, dim_z=128, dim_h=128, dim_hd=512, dim_c=3,
                 dim_x=32, dim_y=32, dim_l=10, multi_discriminator=True,
                 **kwargs):
    '''Initialize the discriminator parameters.
    
    '''

    logger.info('Initializing discriminator parameters.')
    if multi_discriminator:
        logger.info('Forming parameters for multiple scores.')
    else:
        logger.info('Forming parameters for single score.')
    
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
        '''
        if multi_discriminator:
            name_out = 'd_conv_out_{}'.format(level)
            p = param_init_convlayer(
                options={}, params=p, prefix=name_out, nin=dim_out, nout=1,
                kernel_len=5, batch_norm=False)
        '''

        dim_in = dim_out
    
    '''
    p = param_init_fflayer(
        options={}, params=p, prefix='d_ff_x',
        nin=dim_h * scale * dim_x // scale * dim_y // scale // 2, nout=dim_h,
        ortho=False, batch_norm=False)
    '''
        
    p = param_init_fflayer(
        options={}, params=p, prefix='d_ff_y', nin=dim_l, nout=dim_h,
        ortho=False, batch_norm=False)

    for level in xrange(2):
        name = 'd_ff_{}'.format(level)

        if level == 0:
            #dim_in = 2 * dim_h + dim_z
            dim_in = dim_h * scale * dim_x // scale * dim_y // scale // 2 + dim_h + dim_z
            
        if level == 1:
            dim_out = 1
        else:
            dim_out = dim_hd
            
        logger.debug('Discriminator ff dim in / out: {}-{}'.format(
            dim_in, dim_out))
        
        p = param_init_fflayer(
            options={}, params=p, prefix=name, nin=dim_in, nout=dim_out,
            ortho=False, batch_norm=False)

        if multi_discriminator or level == 1:
            name_out = 'd_ff_out_{}'.format(level)
            p = param_init_fflayer(
                options={}, params=p, prefix=name_out, nin=dim_out, nout=1,
                ortho=False, batch_norm=False)

        dim_in = dim_out
    
    return init_tparams(p)


# MODELS #######################################################################

def z_to_x(p, z, n_levels=3, dim_h=128, dim_x=32, dim_y=32, dim_l=10,
           return_tensors=False, noise_std=1.,
           nonlinearity='lambda x: T.tanh(x)', **kwargs):
    '''z to x transition.
    
    '''
    scale = 2 ** n_levels
    d = OrderedDict()
    
    logger.debug("Added extra noise input")
    z = T.concatenate([z, srng.normal(std=noise_std, size=z.shape)], axis=1)
    d['z'] = z
    x = fflayer(
        tparams=p, state_below=z, options={}, prefix='z_x_ff',
        activ='lambda x: tensor.nnet.relu(x, alpha=0.02)')
    d['x0'] = x

    x = x.reshape((-1, dim_h * scale // 2, dim_x // scale, dim_y // scale))
    d['x0_rs'] = x

    for level in xrange(n_levels):
        name = 'z_x_conv_{}'.format(level)
        if level == n_levels - 1:
            activ = nonlinearity
        else:
            activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'
        x = convlayer(
            tparams=p, state_below=x, options={}, prefix=name,
            activ=activ, stride=-2)
        d['x{}'.format(level + 1)] = x
    d['x_f'] = x
        
    y = z
    for level in xrange(2):
        name = 'z_y_ff_{}'.format(level)
        if level == 1:
            activ = 'lambda x: x'
        else:
            activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'
        y = fflayer(
            tparams=p, state_below=y, options={}, prefix=name, activ=activ)
        d['y{}'.format(level + 1)] = y  
    d['y_f'] = y

    if return_tensors:
        return d
    else:
        return x, y


def x_to_z(p, x, y, n_levels=3, dim_z=128, dim_h=128, dim_c=3, dim_x=32,
           dim_y=32, dim_l=10, normalize_z=False, return_tensors=False,
           noise_std=1., **kwargs):
    '''x to z transition.
    
    '''
    out = OrderedDict()
    h_x = x.reshape((-1, dim_c, dim_x, dim_y))
    epsilon = srng.normal(std=noise_std,
                          size=(x.shape[0], dim_z), dtype=x.dtype)
    
    h_e = fflayer(tparams=p, state_below=epsilon, options={}, prefix='n_z_ff_x',
                  activ='lambda x: tensor.nnet.relu(x, alpha=0.02)')
    h_e = h_e.reshape((-1, dim_h, dim_x / 2, dim_y / 2))
    
    for level in xrange(n_levels):
        if level == 1:
            h_x = h_x + h_e
        name = 'x_z_conv_{}'.format(level)
        activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'
        h_x = convlayer(
            tparams=p, state_below=h_x, options={}, prefix=name, activ=activ,
            stride=2)
        out['h_x{}'.format(level)] = h_x
    h_x = h_x.flatten(2)
    
    h_x = fflayer(
        tparams=p, state_below=h_x, options={}, prefix='x_z_final1',
        activ='lambda x: tensor.nnet.relu(x, alpha=0.02)')
    out['h_x_f'] = h_x
    
    h_e = fflayer(tparams=p, state_below=epsilon, options={}, prefix='n_z_ff_y',
                  activ='lambda x: tensor.nnet.relu(x, alpha=0.02)')
    h_y = y
    
    for level in xrange(2):
        if level == 1:
            h_y = h_y + h_e
        name = 'y_z_ff_{}'.format(level)
        activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'
        h_y = fflayer(
            tparams=p, state_below=h_y, options={}, prefix=name, activ=activ)
        out['h_y{}'.format(level)] = h_y
    
    h = T.concatenate([h_y, h_x], axis=1)

    logger.debug('Forming layer with name {}'.format('x_z_final2'))
    z = fflayer(
        tparams=p, state_below=h, options={}, prefix='x_z_final2',
        #activ='lambda x: x'
        activ='lambda x: T.tanh(x)'
    )
    out['z_f'] = z

    if return_tensors:
        return out
    else:
        return z


def discriminator(p, x, y, z, n_levels=3, dim_z=128, dim_h=128, dim_hd=512,
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
        h = convlayer(
            tparams=p, state_below=h, options={}, prefix=name, activ=activ,
            stride=2)
        outs['h_conv_{}'.format(level)] = h
        '''
        if multi_discriminator:
            name_out = 'd_conv_out_{}'.format(level)
            d = convlayer(
                tparams=p, state_below=h, options={}, prefix=name_out,
                activ='lambda x: x', stride=2)
            ds.append(d)
            outs['d_conv_{}'.format(level)] = d
        '''
            
    h_y = fflayer(
        tparams=p, state_below=y, options={}, prefix='d_ff_y', activ=activ,
        mean_ln=False)
    
    h = h.flatten(2)
    '''
    h_x = fflayer(
        tparams=p, state_below=h, options={}, prefix='d_ff_x', activ=activ,
        mean_ln=False
    )
    '''
    outs['hf'] = h
    outs['z'] = z
    
    h = T.concatenate([z, h, h_y], axis=1)
    outs['h1'] = h

    for level in xrange(2):
        name = 'd_ff_{}'.format(level)
        h = fflayer(
            tparams=p, state_below=h, options={}, prefix=name, activ=activ,
            mean_ln=False)
        outs['h_ff_{}'.format(level)] = h
        if multi_discriminator or level == 1:
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
