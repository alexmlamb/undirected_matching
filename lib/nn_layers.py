import theano
from theano import tensor
import numpy
from utils import ortho_weight , norm_weight, init_tparams #uniform_weight

import settings
profile = settings.profile


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer')
         }

def _p(pp, name):
    return '%s_%s' % (pp, name)


def prfx(pp, name):
    return '%s_%s' % (pp, name)


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def zero_vector(length):
    return numpy.zeros((length, )).astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def relu(x):
    return theano.maximum(0.0, x)


def ln(x, b, s):
     _eps = 1e-5
     output = (x - x.mean(1)[:,None]) / tensor.sqrt((x.var(1)[:,None] + _eps))
     output = s[None, :] * output + b[None,:]
     return output



# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options,
                       params,
                       prefix='ff',
                       nin=None,
                       nout=None,
                       ortho=True,
                       batch_norm=False):

    params[prefix + '_W'] = norm_weight(nin, nout)
    params[prefix + '_b'] = zero_vector(nout)

    return params


def fflayer(tparams,
            state_below,
            options,
            prefix='rconv',
            activ='lambda x: tensor.tanh(x)',
            batch_norm=False,
            layer_norm=False,
            mean_ln=False,
            mean_bn=False,
            **kwargs):
    preactivation = tensor.dot(state_below, tparams[prefix + '_W']) +tparams[prefix + '_b']

    if batch_norm:
        preactivation = (preactivation - preactivation.mean(axis=0)) / (0.0001 + preactivation.std(axis=0))
    if layer_norm:
        preactivation = (preactivation - preactivation.mean(axis=1,keepdims=True)) / (0.0000001 + preactivation.std(axis=1,keepdims=True))
    if mean_bn:
        preactivation = (preactivation - preactivation.mean(axis=0)) + tparams[prefix + '_b']
    if mean_ln:
        preactivation = (preactivation - preactivation.mean(axis=1,keepdims=True)) + tparams[prefix + '_b']

    return eval(activ)(preactivation)

# GRU layer
def param_init_gru(options, param, prefix='gru', nin=None, dim=None):

    param[prefix + '_W'] = numpy.concatenate(
        [
            norm_weight(nin, dim), norm_weight(nin, dim)
        ],
        axis=1)
    param[prefix + '_U'] = numpy.concatenate(
        [
            ortho_weight(dim), ortho_weight(dim)
        ],
        axis=1)
    param[prefix + '_b'] = zero_vector(2 * dim)

    param[prefix + '_Wx'] = norm_weight(nin, dim)
    param[prefix + '_Ux'] = ortho_weight(dim)
    param[prefix + '_bx'] = zero_vector(dim)

    return param


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]


def gru_layer(tparams,
              state_below,
              options,
              prefix='gru',
              mask=None,
              one_step=False,
              init_state=None,
              backwards=False,
              **kwargs):
    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = state_below.shape[0]

    dim = tparams[prefix + '_Ux'].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # state_below is the input word embeddings
    state_below_ = (tensor.dot(state_below, tparams[prefix + '_W']) +
                    tparams[prefix + '_b'])
    state_belowx = (tensor.dot(state_below, tparams[prefix + '_Wx']) +
                    tparams[prefix + '_bx'])

    def _step_slice(m_, x_, xx_,  h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice
    shared_vars = [tparams[prefix + '_U'], tparams[prefix + '_Ux']]
    # set initial state to all zeros
    if init_state is None:
        init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)
    else:
        init_state = init_state

    if one_step:  # sampling
        rval = _step(*(seqs+[init_state]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[init_state],
                                non_sequences=shared_vars,
                                name=prefix + '_layers',
                                n_steps=nsteps,
                                profile=profile,
                                strict=True,
                                go_backwards=backwards)
    rval = [rval]
    return rval



# LN-GRU layer
def param_init_lngru(options, params, prefix='lngru', nin=None, dim=None):
    """
     Gated Recurrent Unit (GRU) with LN
    """
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    W = numpy.concatenate([norm_weight(nin,dim),
                            norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W.astype('float32')
    params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                            ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U.astype('float32')

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx.astype('float32')
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux.astype('float32')
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

     # LN parameters
    scale_add = 0.0
    scale_mul = 1.0
    params[_p(prefix,'b1')] = scale_add * numpy.ones((2*dim)).astype('float32')
    params[_p(prefix,'b2')] = scale_add * numpy.ones((1*dim)).astype('float32')
    params[_p(prefix,'b3')] = scale_add * numpy.ones((2*dim)).astype('float32')
    params[_p(prefix,'b4')] = scale_add * numpy.ones((1*dim)).astype('float32')
    params[_p(prefix,'s1')] = scale_mul * numpy.ones((2*dim)).astype('float32')
    params[_p(prefix,'s2')] = scale_mul * numpy.ones((1*dim)).astype('float32')
    params[_p(prefix,'s3')] = scale_mul * numpy.ones((2*dim)).astype('float32')
    params[_p(prefix,'s4')] = scale_mul * numpy.ones((1*dim)).astype('float32')

    return params

def lngru_layer(tparams, state_below, options, prefix='lngru', init_state=None, mask=None, one_step=False, **kwargs):


     """
     Feedforward pass through GRU with LN
     def gru_layer(tparams, state_below, options, prefix='gru',
                   mask=None, one_step=False, init_state=None, **kwargs):
     """
     nsteps = state_below.shape[0]
     if state_below.ndim == 3:
         n_samples = state_below.shape[1]
     else:
         n_samples = 1

     dim = tparams[_p(prefix,'Ux')].shape[1]

     if init_state == None:
         init_state = tensor.alloc(0., n_samples, dim)

     if mask == None:
         mask = tensor.alloc(1., state_below.shape[0], 1)

     def _slice(_x, n, dim):
         if _x.ndim == 3:
             return _x[:, :, n*dim:(n+1)*dim]
         return _x[:, n*dim:(n+1)*dim]

     state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
     state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
     #U = tparams[_p(prefix, 'U')]
     #Ux = tparams[_p(prefix, 'Ux')]

     def _step_slice(m_, x_, xx_, h_, U, Ux, b1, b2, b3, b4, s1, s2, s3, s4):

         x_ = ln(x_, b1, s1)
         xx_ = ln(xx_, b2, s2)

         preact = tensor.dot(h_, U)
         preact = ln(preact, b3, s3)
         preact += x_

         r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
         u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

         preactx = tensor.dot(h_, Ux)
         preactx = ln(preactx, b4, s4)
         preactx = preactx * r
         preactx = preactx + xx_

         h = tensor.tanh(preactx)

         h = u * h_ + (1. - u) * h
         h = m_[:,None] * h + (1. - m_)[:,None] * h_

         return h

     seqs = [mask, state_below_, state_belowx]
     _step = _step_slice

     non_seqs = [tparams[_p(prefix, 'U')], tparams[_p(prefix, 'Ux')]]
     non_seqs += [tparams[_p(prefix, 'b1')], tparams[_p(prefix, 'b2')], tparams[_p(prefix, 'b3')], tparams[_p(prefix, 'b4')]]
     non_seqs += [tparams[_p(prefix, 's1')], tparams[_p(prefix, 's2')], tparams[_p(prefix, 's3')], tparams[_p(prefix, 's4')]]

     if one_step:
         #rval = _step(*(seqs+[init_state, tparams[_p(prefix, 'U')], tparams[_p(prefix, 'Ux')]]))
         rval = _step(*(seqs+[init_state] + non_seqs))
     else:
         rval, updates = theano.scan(_step,
                                     sequences=seqs,
                                     outputs_info = [init_state],
                                     non_sequences = non_seqs,
                                     name=_p(prefix, '_layers'),
                                     n_steps=nsteps,
                                     profile=False,
                                     strict=True)
     rval = [rval]
     return rval

def param_init_lstm(options,
                     params,
                     prefix='lstm',
                     nin=None,
                     dim=None):
     if nin is None:
         nin = options['dim_proj']

     if dim is None:
         dim = options['dim_proj']

     W = numpy.concatenate([norm_weight(nin,dim),
                            norm_weight(nin,dim),
                            norm_weight(nin,dim),
                            norm_weight(nin,dim)],
                            axis=1)

     params[prfx(prefix,'W')] = W
     U = numpy.concatenate([ortho_weight(dim),
                            ortho_weight(dim),
                            ortho_weight(dim),
                            ortho_weight(dim)],
                            axis=1)

     params[prfx(prefix,'U')] = U
     params[prfx(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

     return params

def lstm_layer(tparams, state_below,
                options,
                prefix='lstm',
                mask=None, one_step=False,
                init_state=None,
                init_memory=None,
                nsteps=None,
                **kwargs):

     if nsteps is None:
         nsteps = state_below.shape[0]

     if state_below.ndim == 3:
         n_samples = state_below.shape[1]
     else:
         n_samples = 1

     param = lambda name: tparams[prfx(prefix, name)]
     dim = param('U').shape[0]

     if mask is None:
         mask = tensor.alloc(1., state_below.shape[0], 1)

     # initial/previous state
     if init_state is None:
         if not options['learn_h0']:
             init_state = tensor.alloc(0., n_samples, dim)
         else:
             init_state0 = theano.shared(numpy.zeros((options['dim'])),
                                  name=prfx(prefix, "h0"))
             init_state = tensor.alloc(init_state0, n_samples, dim)
             tparams[prfx(prefix, 'h0')] = init_state0

     U = param('U')
     b = param('b')
     W = param('W')
     non_seqs = [U, b, W]

     # initial/previous memory
     if init_memory is None:
         init_memory = tensor.alloc(0., n_samples, dim)

     def _slice(_x, n, dim):
         if _x.ndim == 3:
             return _x[:, :, n*dim:(n+1)*dim]
         return _x[:, n*dim:(n+1)*dim]

     def _step(mask, sbelow, sbefore, cell_before, *args):
         preact = tensor.dot(sbefore, param('U'))
         preact += sbelow
         preact += param('b')

         i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
         f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
         o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
         c = tensor.tanh(_slice(preact, 3, dim))

         c = f * cell_before + i * c
         c = mask * c + (1. - mask) * cell_before
         h = o * tensor.tanh(c)
         h = mask * h + (1. - mask) * sbefore

         return h, c

     lstm_state_below = tensor.dot(state_below, param('W')) + param('b')
     if state_below.ndim == 3:
         lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                      state_below.shape[1],
                                                      -1))
     if one_step:
         mask = mask.dimshuffle(0, 'x')
         h, c = _step(mask, lstm_state_below, init_state, init_memory)
         rval = [h, c]
     else:
         if mask.ndim == 3 and mask.ndim == state_below.ndim:
             mask = mask.reshape((mask.shape[0], \
                                  mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
         elif mask.ndim == 2:
             mask = mask.dimshuffle(0, 1, 'x')

         rval, updates = theano.scan(_step,
                                     sequences=[mask, lstm_state_below],
                                     outputs_info = [init_state,
                                                     init_memory],
                                     name=prfx(prefix, '_layers'),
                                     non_sequences=non_seqs,
                                     strict=True,
                                     n_steps=nsteps)
     return rval


# LN-LSTM init
def param_init_lnlstm(options,
                     params,
                     prefix='lnlstm',
                     nin=None,
                     dim=None):
     if nin is None:
         nin = options['dim_proj']

     if dim is None:
         dim = options['dim_proj']

     W = numpy.concatenate([norm_weight(nin,dim),
                            norm_weight(nin,dim),
                            norm_weight(nin,dim),
                            norm_weight(nin,dim)],
                            axis=1)

     params[prfx(prefix,'W')] = W
     U = numpy.concatenate([ortho_weight(dim),
                            ortho_weight(dim),
                            ortho_weight(dim),
                            ortho_weight(dim)],
                            axis=1)

     params[prfx(prefix,'U')] = U
     params[prfx(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

     # lateral parameters
     scale_add = 0.0
     scale_mul = 1.0
     params[prfx(prefix,'b1')] = scale_add * numpy.ones((4*dim)).astype('float32')
     params[prfx(prefix,'b2')] = scale_add * numpy.ones((4*dim)).astype('float32')
     params[prfx(prefix,'b3')] = scale_add * numpy.ones((1*dim)).astype('float32')
     params[prfx(prefix,'s1')] = scale_mul * numpy.ones((4*dim)).astype('float32')
     params[prfx(prefix,'s2')] = scale_mul * numpy.ones((4*dim)).astype('float32')
     params[prfx(prefix,'s3')] = scale_mul * numpy.ones((1*dim)).astype('float32')

     return params


# LN-LSTM layer
def lnlstm_layer(tparams, state_below,
                  options,
                  prefix='lnlstm',
                  mask=None, one_step=False,
                  init_state=None,
                  init_memory=None,
                  nsteps=None,
                  **kwargs):

     if nsteps is None:
         nsteps = state_below.shape[0]

     if state_below.ndim == 3:
         n_samples = state_below.shape[1]
     else:
         n_samples = 1

     param = lambda name: tparams[prfx(prefix, name)]
     dim = param('U').shape[0]

     if mask is None:
         mask = tensor.alloc(1., state_below.shape[0], 1)

     # initial/previous state
     if init_state is None:
         #if not options['learn_h0']:
         if False:#not options['learn_h0']:
             init_state = tensor.alloc(0., n_samples, dim)
         else:
             init_state0 = theano.shared(numpy.zeros((options['dim'])),
                                  name=prfx(prefix, "h0"))
             init_state = tensor.alloc(init_state0, n_samples, dim)
             tparams[prfx(prefix, 'h0')] = init_state0

     U = param('U')
     b = param('b')
     W = param('W')
     non_seqs = [U, b, W]
     non_seqs.extend(list(map(param, "b1 b2 b3 s1 s2 s3".split())))

     # initial/previous memory
     if init_memory is None:
         init_memory = tensor.alloc(0., n_samples, dim)

     def _slice(_x, n, dim):
         if _x.ndim == 3:
             return _x[:, :, n*dim:(n+1)*dim]
         return _x[:, n*dim:(n+1)*dim]

     def _step(mask, sbelow, sbefore, cell_before, *args):
         sbelow_ = ln(sbelow, param('b1'), param('s1'))
         sbefore_ = ln(tensor.dot(sbefore, param('U')), param('b2'), param('s2'))

         preact = sbefore_ + sbelow_ + param('b')

         i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
         f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
         o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
         c = tensor.tanh(_slice(preact, 3, dim))

         c = f * cell_before + i * c
         c = mask * c + (1. - mask) * cell_before

         c_ = ln(c, param('b3'), param('s3'))
         h = o * tensor.tanh(c_)
         h = mask * h + (1. - mask) * sbefore

         return h, c

     lstm_state_below = tensor.dot(state_below, param('W')) + param('b')
     if state_below.ndim == 3:
         lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                      state_below.shape[1],
                                                      -1))
     if one_step:
         mask = mask.dimshuffle(0, 'x')
         h, c = _step(mask, lstm_state_below, init_state, init_memory)
         rval = [h, c]
     else:
         if mask.ndim == 3 and mask.ndim == state_below.ndim:
             mask = mask.reshape((mask.shape[0], \
                                  mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
         elif mask.ndim == 2:
             mask = mask.dimshuffle(0, 1, 'x')

         rval, updates = theano.scan(_step,
                                     sequences=[mask, lstm_state_below],
                                     outputs_info = [init_state,
                                                     init_memory],
                                     name=prfx(prefix, '_layers'),
                                     non_sequences=non_seqs,
                                     strict=True,
                                     n_steps=nsteps)
     return rval






if __name__ == "__main__":
    print "setting up gru"

    params = {}
    gru_params = param_init_gru({}, param=params, prefix='gru', nin=1024, dim=1024)

    tparams = init_tparams(params)

    import numpy as np

    #h_out = gru_layer(tparams,state_below,options={},prefix='gru',mask=None,one_step=True,init_state=None,backwards=False)



