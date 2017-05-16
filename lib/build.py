'''Compile functions.

'''

from os import path
import logging

import numpy as np
import numpy.random as rng
import theano
from theano import tensor as T

from loss import bgan_loss, lsgan_loss, wgan_loss
from viz import plot_chain, plot_images


logger = logging.getLogger('UDGAN.build')
consider_constant = theano.gradient.disconnected_grad # changed from orginal
floatX = theano.config.floatX

ARCH = None


# COMPILE ######################################################################

def compile_train(updates, inputs, outputs):
    logger.info("Compiling train function.")
    inputs = [inputs[k] for k in ['X', 'Y', 'Z']]
    f = theano.function(
        inputs, outputs=outputs, updates=updates,
        on_unused_input='ignore')
    return f

# COMPILE ######################################################################

def compile_generation(X=None, Z=None, outputs=None):
    logger.info("Compiling generation function.")
    inputs = []
    if X is not None: inputs.append(X)
    if Z is not None: inputs.append(Z)
    f = theano.function(inputs, outputs=outputs,
        on_unused_input='ignore')
    return f


def compile_chain(X=None, Y=None, Z=None, n_steps=None,
                  fix_X=False, fix_Y=False, fix_half_X=False,
                  **model_args):
    logger.info("Compiling chain function.")
    Xs, Ys, Ps, Zs = p_chain(X=X, Y=Y, Z=Z, num_iterations=n_steps,
                             fix_X=fix_X, fix_Y=fix_Y, fix_half_X=fix_half_X,
                             **model_args)
    
    inputs = []
    if X is not None: inputs.append(X)
    if Y is not None: inputs.append(Y)
    if Z is not None: inputs.append(Z)
    
    f = theano.function(inputs, outputs=dict(
        Xs=T.as_tensor_variable(Xs), Ys=T.as_tensor_variable(Ys),
        Zs=T.as_tensor_variable(Zs)))
    
    return f


# VIZ ##########################################################################

class Visualizer(object):
    def __init__(self, X=None, Y=None, Z=None, outputs=None,
                 viz_type='simple', extra_args=None, name='gen',
                 out_dir=None, archive=False, **kwargs):
        logger.info('Making vizualizer with type {}'.format(viz_type))
        self.viz_type = viz_type
        self.extra_args = extra_args or {}
        self.sources = []
        if out_dir is None: raise ValueError('out_path not set.')
        self.out_dir = out_dir
        self.name = name
        self.archive = archive
        if self.viz_type == 'simple':
            self.fn = compile_generation(X=X, Z=Z, outputs=outputs)
            if X is not None:
                self.sources.append('X')
            if Z is not None:
                self.sources.append('Z')
        elif self.viz_type == 'chain':
            self.fn = compile_chain(X=X, Y=Y, Z=Z, **kwargs)
            if X is not None:
                self.sources.append('X')
            if Y is not None:
                self.sources.append('Y')
            if Z is not None:
                self.sources.append('Z')
        else:
            raise NotImplementedError(self.viz_type)
    
    def run(self, inputs, suffix=''):
        suffix = str(suffix)
        inputs_ = [inputs[k] for k in self.sources]
        outs = self.fn(*inputs_)
        
        kwargs = {}
        kwargs.update(**self.extra_args)
        if suffix is None: suffix = ''
        suffix = ('_{}'.format(suffix)) if len(suffix) > 0 else ''
        file_path = path.join(self.out_dir, '{}{}'.format(self.name, suffix))
        if self.viz_type == 'simple':
            file_path += '.png'
            for k in outs.keys():
                v = outs[k]
                if len(v.shape) > 3:
                    outs[k] = (v + 1.) * 0.5
            kwargs.update(**outs)
            plot_images(file_path=file_path, **kwargs)
        elif self.viz_type == 'chain':
            file_path += '.gif'
            x = (outs['Xs'] + 1.) * 128.
            plot_chain(x, file_path=file_path, **kwargs)


def add_viz(**kwargs):
    return Visualizer(**kwargs)

# GRAPH ########################################################################

def p_chain(X=None, Y=None, Z=None, num_iterations=None, pd_steps=None,
            fix_X=False, fix_Y=False, fix_half_X=False, **model_args):
    '''P chain
    
    .. note::
        the returned ylst are pre-sampling. This was done to do multiple
        samples later
    
    '''
    z_to_x = ARCH.z_to_x
    x_to_z = ARCH.x_to_z

    if X is not None:
        logger.info('X is given')
        xlst = [X]
    else:
        xlst = []
    
    if Z is not None:
        logger.info('Z is given')
        zlst = [Z]
    else:
        zlst = []
    
    if Y is not None:
        logger.info('Y is given')
        ylst = [Y]
        plst = [Y]
    else:
        ylst = []
        plst = []
    
    if fix_X or fix_half_X:
        if fix_half_X:
            logger.info('Inpainting half of X')
        else:
            logger.info('X is fixed')
        X_ = X
    if fix_Y:
        logger.info('Y is fixed')
        Y_ = Y
    if fix_X and fix_Y:
        raise ValueError()
    
    if X is not None:
        if ARCH._semi_supervised and Y is not None:            
            Z = x_to_z(X, Y, **model_args)
        elif not ARCH._semi_supervised:
            Z = x_to_z(X, **model_args)
        zlst.append(Z)

    for i in xrange(num_iterations):
        #out = z_to_x(consider_constant(Z), **model_args)
        out = z_to_x(Z, **model_args)
        
        if ARCH._semi_supervised:
            X, P = out
            plst.append(P)
            Y = sample_multinomial(P)
            ylst.append(Y)            
            if fix_Y:
                Y = Y_
        else:
            X = out
            
        if fix_X:
            X = X_
        if fix_half_X:
            dim_x = X.shape[2]
            X = T.set_subtensor(X[:, :, :dim_x // 2, :],
                                X_[:, :, :dim_x // 2, :])
        
        xlst.append(X)

        if i < num_iterations - 1:
            if ARCH._semi_supervised:
                if i == num_iterations - pd_steps - 1:
                    Z = x_to_z(consider_constant(X), Y, **model_args)
                else:
                    Z = x_to_z(X, Y, **model_args)
            else:
                if i == num_iterations - pd_steps - 1:
                    Z = x_to_z(consider_constant(X), **model_args)
                else:
                    Z = x_to_z(X, **model_args)
            
            zlst.append(Z)

    assert len(xlst) == len(zlst)
    return xlst, ylst, plst, zlst


def q_chain(X, Y, num_iterations, test=False, **model_args):
    x_to_z = ARCH.x_to_z
    xlst = [X]
    ylst = [Y]
    zlst = []
    
    if ARCH._semi_supervised:
        new_z = x_to_z(X, Y)
    else:
        new_z = x_to_z(X, **model_args)
        
    zlst.append(new_z)

    return xlst, ylst, zlst


def initialize_params(**model_args):
    logger.info('Initializing parameters from {} with args {}'.format(
        ARCH.__name__, model_args))
    gparams = ARCH.init_gparams(**model_args)
    dparams = ARCH.init_dparams(**model_args)
    
    return gparams, dparams


def make_vars(**data_shapes):
    logger.info('Setting input variables for {}'.format(data_shapes.keys()))
    inputs = {}
    t_dict = {
        1:T.vector,
        2:T.matrix,
        3:T.tensor3,
        4:T.tensor4
    }
    k_dict = {
        'features': 'X',
        'targets': 'Y',
        'captions': 'Y'
    }
    k_map = {}
    for k, v in data_shapes.items():
        x = t_dict[len(v)](k)
        inputs[k_dict[k]] = x
        k_map[k_dict[k]] = k
    inputs['Z'] = T.matrix('latents')
    return inputs, k_map


def build_model(X=None, Y=None, Z=None, num_steps=None, pd_steps=None,
                start_on_x=None, **model_args):
    logger.info('Building graph')
    if start_on_x:
        logger.info('Starting chain at data.')
        X_ = X
    else:
        logger.info('Starting chain at gaussian noise.')
        X_ = None
    p_xs, p_ys, p_ps, p_zs = p_chain(Z=Z, X=X_, num_iterations=num_steps,
                                     pd_steps=pd_steps, **model_args)
    q_xs, q_ys, q_zs = q_chain(X, Y, num_steps, **model_args)

    logger.debug("p chain x: {}".format(p_xs))
    logger.debug("p chain p: {}".format(p_ps))
    logger.debug("p chain z: {}".format(p_zs))
    logger.debug("q chain x: {}".format(q_xs))
    logger.debug("q chain y: {}".format(q_ys))
    logger.debug("q chain z: {}".format(q_zs))
    
    outs = dict(p_xs=p_xs, p_ps=p_ps, p_zs=p_zs,
                q_xs=q_xs, q_ys=q_ys, q_zs=q_zs)
    
    return outs


def test(inputs, datasets, num_steps=None, pd_steps=None, loss=None,
         n_samples=None, start_on_x=None, **model_args):
    '''Test model and graph.
    
    '''
    
    d = ARCH.z_to_x(inputs['Z'], return_tensors=True, **model_args)
    d.update(ARCH.x_to_z(inputs['X'], return_tensors=True,
                           **model_args))
    d.update(ARCH.discriminator(inputs['X'], inputs['Z'], return_tensors=True,
                                **model_args))
    
    x, y = datasets['train'].get_epoch_iterator().next()
    z = rng.normal(size=(x.shape[0], model_args['dim_z'])).astype(floatX)
    
    inputs = [inputs[k_] for k_ in ['X', 'Y', 'Z']]
    for k, v in d.items():
        print 'Trying {}'.format(k)
        f = theano.function(inputs, v, on_unused_input='ignore')
        print f(x, y, z).shape
    assert False, d


def set_arch(arch):
    global ARCH
    logger.info('Setting model architecture to {}'.format(arch.__name__))
    ARCH = arch


def unlabeled_results(p_xs=None, p_ps=None, p_zs=None,
                      q_xs=None, q_ys=None, q_zs=None,
                      pd_steps=None, loss=None, **model_args):
    logger.info('Using {} steps of p in discriminator'.format(pd_steps))
    
    D_qs, _ = ARCH.discriminator(q_xs[-1], q_zs[-1], **model_args)
    
    D_ps = []
    for i in xrange(pd_steps):
        D_ps_, _ = ARCH.discriminator(
            p_xs[-(i + 1)], p_zs[-(i + 1)], **model_args)
        D_ps += D_ps_
    
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
    
    dloss, gloss = loss_fn(D_qs, D_ps)
    
    results = {
        'D loss': dloss,
        'G loss': gloss,
        #'D_q': T.mean(D_qs),
        #'D_p': T.mean(D_ps),
        #'Q x': q_xs[-1].mean(),
        #'Q z': q_zs[-1].mean(),
        #'P x': p_xs[-1].mean(),
        #'P z': p_zs[-1].mean(),
        'p(real)': (q_xs[-1] > 0.5).mean(),
        'p(fake)': (p_xs[-1] < 0.5).mean()
    }
    
    return results