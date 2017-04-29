'''Convolutional model

'''

from nn_layers import param_init_fflayer, param_init_convlayer


# INITIALIZE PARAMETERS ########################################################

def init_gparams(p, n_levels=3, dim_z=128, dim_h=128, dim_c=3, dim_x=32,
                 dim_y=32):
    scale = 2 ** (n_levels)
    dim_in = dim_z * 2
    dim_out = dim_h
    
    p = param_init_fflayer(
        options={}, params=p, prefix='z_x_ff', nin=dim_in,
        nout=dim_out * dim_x * dim_y // scale, ortho=False, batch_norm=True)

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
        nin=dim_h * dim_x * dim_y // scale, nout=dim_z, ortho=False,
        batch_norm=False)

    p = param_init_fflayer(
        options={}, params=p, prefix='x_z_logsigma',
        nin=dim_h * dim_x * dim_y // scale, nout=dim_z, ortho=False,
        batch_norm=False)

    return init_tparams(p)


def init_dparams(p, n_levels=3, dim_z=128, dim_h=128, dim_hd=512, dim_c=3,
                 dim_x=32, dim_y=32, multi_discriminator=True):
    dim_in = dim_c
    scale = 2 ** (n_levels)

    for level in xrange(n_levels):
        name = 'd_conv_{}'.format(level)
        name_out = 'd_conv_out_{}'.format(level)
        
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
            p = param_init_convlayer(
                options={}, params=p, prefix=name_out, nin=dim_out, nout=1,
                kernel_len=5, batch_norm=False)

        dim_in = dim_out

    for level in xrange(0, n_levels):
        name = 'd_ff_{}'.format(level)
        name_out = 'd_ff_out_{}'.format(level)

        if level == 0:
            dim_in = dim_z + dim_h * dim_x * dim_y // scale
        
        if level == n_levels - 1:
            dim_out = 1
        else:
            dim_out = dim_hg
        
        p = param_init_ff_layer(
            options={}, params=p, prefix=name, nin=dim_in, nout=dim_out,
            ortho=False, batch_norm=False)

        if multi_discriminator or level == n_levels - 1:
            p = param_init_fflayer(
                options={}, params=p, prefix=name_out, nin=dim_out, nout=1,
                ortho=False, batch_norm=False)

        dim_in = dim_out
    
    return init_tparams(p)


# MODELS #######################################################################

def z_to_x(p, z):
    logger.info("Added extra noise input")
    z = join2(z, srng.normal(size=z.shape))

    x = fflayer(
        tparams=p, state_below=z, options={}, prefix='z_x_ff',
        activ='lambda x: tensor.nnet.relu(x, alpha=0.02)')

    x = x.reshape((batch_size, dim_h, dim_x // scale, dim_y // scale))

    for level in xrange(n_levels):
        name = 'z_x_conv_{}'.format(level)
        
        if level == n_levels - 1:
            activ = 'lambda x: x'
        else:
            activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'

        x = convlayer(
            tparams=p, state_below=x, options={}, prefix=name,
            activ=activ, stride=-2)

    x = x.flatten(2)

    return x


def x_to_z(p, x):
    h = x.reshape((batch_size, dim_c, dim_x, dim_y))

    for level in xrange(n_levels):
        name = 'x_z_conv_{}'.format(level)
        activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'
        h = convlayer(
            tparams=p, state_below=h, options={}, prefix=name, activ=activ,
            stride=2)

    h = h.flatten(2)

    log_sigma = fflayer(
        tparams=p, state_below=h, options={}, prefix='x_z_logsigma',
        activ='lambda x: x')

    mu = fflayer(
        tparams=p, state_below=h, options={}, prefix='x_z_mu',
        activ='lambda x: x')

    eps = srng.normal(size=sigma.shape)
    z = eps * T.exp(log_sigma) + mu

    if normalize_z:
        z = (z - T.mean(z, axis=0, keepdims=True)) / (
            epsilon + T.std(z, axis=0, keepdims=True))

    return z


def discriminator(p, x, z):
    h = x.reshape((batch_size, dim_c, dim_x, dim_y))
    activ = 'lambda x: tensor.nnet.relu(x, alpha=0.02)'

    for level in xrange(n_levels):
        h = convlayer(
            tparams=p, state_below=h, options={},
            prefix='DC_1', activ=activ, stride=2)

    dc_2 = convlayer(
        tparams=p, state_below=dc_1, options={}, prefix='DC_2',
        activ='lambda x: tensor.nnet.relu(x, alpha=0.02)', stride=2)

    dc_3 = convlayer(
        tparams=p, state_below=dc_2, options={}, prefix='DC_3',
        activ='lambda x: tensor.nnet.relu(x, alpha=0.02)', stride=2)

    inp = join2(z, dc_3.flatten(2))

    h1 = fflayer(
        tparams=p, state_below=inp, options={}, prefix='D_1',
        activ='lambda x: tensor.nnet.relu(x, alpha=0.02)', mean_ln=False)

    h2 = fflayer(
        tparams=p, state_below=h1, options={}, prefix='D_2',
        activ='lambda x: tensor.nnet.relu(x, alpha=0.02)', mean_ln=False)

    h3 = fflayer(
        tparams=p, state_below=h2, options={}, prefix='D_3',
        activ='lambda x: tensor.nnet.relu(x, alpha=0.02)', mean_ln=False)

    D1 = fflayer(
        tparams=p, state_below=h1, options={}, prefix='D_o_1',
        activ='lambda x: x')

    D2 = fflayer(
        tparams=p, state_below=h2, options={}, prefix='D_o_2',
        activ='lambda x: x')

    D3 = fflayer(
        tparams=p, state_below=h3, options={}, prefix='D_o_3',
        activ='lambda x: x')

    D4 = convlayer(
        tparams=p, state_below=dc_1, options={}, prefix='D_o_4',
        activ='lambda x: x', stride=2)

    D5 = convlayer(
        tparams=p, state_below=dc_2, options={}, prefix='D_o_5',
        activ='lambda x: x', stride=2)

    D6 = convlayer(
        tparams=p, state_below=dc_3, options={}, prefix='D_o_6',
        activ='lambda x: x', stride=2)

    logger.info("special thing in D (devon: what does this mean?)")
    return [D1, D2, D3, D4, D5, D6], h3
