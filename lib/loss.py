#!/usr/bin/env python

import sys
sys.path.append("/u/lambalex/DeepLearning/dreamprop/lib")

import theano
import theano.tensor as T

from utils import log_sum_exp, log_sum_exp2


floatX = theano.config.floatX


def cast(inp):
    return T.cast(inp, 'float32')

def crossent(p,y): 
    return -T.mean(T.log(p)[T.arange(y.shape[0]), y])

def nll(p,y,n):
    return -T.sum(T.log(p)*expand(y,n))

def accuracy(p,y):
    return T.mean(cast(T.eq(cast(T.argmax(p, axis = 1)),cast(y))))

def expand(y,n):
    return T.extra_ops.to_one_hot(y, n)


def clip_updates(updates, params):
    new_updates = []
    for (p,u) in updates.items():
        if p in params:
            print "UPDATING P IN PARAMS"
            u = T.clip(u, -0.01, 0.01)
        new_updates.append((p,u))
    return new_updates


def lsgan_loss(D_q_lst, D_p_lst):
    dloss = 0.0
    gloss = 0.0

    max_len = max(len(D_q_lst), len(D_p_lst))

    for i in range(len(D_p_lst)):
        D_p = D_p_lst[i]
        dloss += T.mean(T.sqr(0.0 - D_p))
        gloss += T.mean(T.sqr(1.0 - D_p))

    for i in range(len(D_q_lst)):
        D_q = D_q_lst[i]
        dloss += T.mean(T.sqr(1.0 - D_q))
        gloss += T.mean(T.sqr(0.0 - D_q))

    return dloss / max_len, gloss / max_len


def wgan_loss(D_q_lst, D_p_lst):
    dloss = 0.0
    gloss = 0.0

    #for dloss, push up D_p and push down D_q
    #for gloss, push up D_q and push down D_p

    for i in range(len(D_q_lst)):
        D_q = D_q_lst[i]
        D_p = D_p_lst[i]
        dloss += T.mean(D_q) + T.mean(-1.0 * D_p)
        gloss += T.mean(D_p) + T.mean(-1.0 * D_q)

    return dloss / len(D_q_lst), gloss / len(D_q_lst)


def bgan_loss(D_q_lst, D_p_lst):
    dloss = 0.0
    gloss = 0.0
    
    for D_q, D_p in zip(D_q_lst, D_p_lst):
        dloss += (T.nnet.softplus(-D_q)).mean() + (
        T.nnet.softplus(-D_p)).mean() + D_p.mean()
        gloss += (D_p ** 2).mean() + (D_q ** 2).mean()
        
    return dloss / len(D_q_lst), gloss / len(D_q_lst)


def bgan_loss_2(D_q_lst, D_p_lst, samples, g_output_logit):
    
    dloss = 0.0
    gloss = 0.0
    
    for D_q, D_p in zip(D_q_lst, D_p_lst):
        dloss += (T.nnet.softplus(-D_q)).mean() + (
        T.nnet.softplus(-D_p)).mean() + D_p.mean()
        
        gloss += (D_p ** 2).mean() + (D_q ** 2).mean()
        
        if g_output_logit.ndim == 2:
            log_g = (samples * (g_output_logit - log_sum_exp2(
                g_output_logit, axis=1))[None, :, :]).sum(axis=2)
        elif g_output_logit.ndim == 3:
            log_g = (samples * (g_output_logit - log_sum_exp2(
                g_output_logit, axis=2))[None, :, :, :]).sum(axis=(2, 3))
        else:
            raise ValueError()

        log_N = T.log(D_p.shape[0]).astype(floatX)
        log_Z_est = log_sum_exp(D_p - log_N, axis=0)
        log_w_tilde = D_p - T.shape_padleft(log_Z_est) - log_N
        w_tilde = T.exp(log_w_tilde)
        w_tilde_ = theano.gradient.disconnected_grad(w_tilde)
           
        gloss += -(w_tilde_ * log_g[:, :, None]).sum(0).mean()
        
    return dloss / len(D_q_lst), gloss / len(D_q_lst)


if __name__ == "__main__":

    p = T.matrix()
    y = T.ivector()

    f = theano.function([p,y], [crossent(p,y), nll(p,y)], allow_input_downcast=True)

    print f([[0.9,0.1],[0.1,0.9]], [0,0])
    print f([[0.9,0.1]], [1])
    print f([[0.7,0.3]], [0])
    print f([[0.5,0.5]], [0])
    print f([[0.1,0.9]], [0])


