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
import sys

import lasagne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
from viz import plot_images

matplotlib.use('Agg')

from loggers import logger, set_stream_logger, set_file_logger
from loss import accuracy, crossent, lsgan_loss, wgan_loss
from nn_layers import fflayer, convlayer
from utils import (
    init_tparams, join2, srng, dropout, inverse_sigmoid, join3, merge_images)


SLURM_NAME = os.environ["SLURM_JOB_ID"]
NL = 128 # 128 works for nl
NFG = 512
NFD = 512
NUM_STEPS = 3
TRAIN_CLASSIFIER_SEPARATE = True
LATENT_SPARSE = False
PERSISTENT_P_CHAIN = False
BLENDING_RATE = 0.5


class ConsiderConstant(theano.compile.ViewOp):

    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]
consider_constant = ConsiderConstant()


def load_data(dataset, source):
    logger.info('Loading `{}` dataset from `{}`'.format(dataset, source))
    if dataset == "mnist":
        mn = gzip.open(source)
        train, valid, test = pickle.load(mn)
        trainx, trainy = train
        validx, validy = valid
        testx, testy = test
        m = 784

    elif dataset == "anime":
        from load_file import FileData, normalize, denormalize

        animeData = FileData(source, 32, 64)
        m = 32 * 32 * 3

    elif dataset == "svhn":
        from load_svhn import SvhnData
        from load_file import normalize, denormalize

        svhnData = SvhnData(mb_size=64, segment="train")

        
def p_chain(p, z, num_iterations):
    zlst = [z]
    xlst = []

    if num_iterations == 1:
        new_z, new_x = transition(p, zlst[-1])
        zlst.append(new_z)
        xlst.append(new_x)

    elif num_iterations == 3:

        logger.info("DOING 3 steps!")

        new_x = z_to_x(p, zlst[-1])
        xlst.append(new_x)
        new_z = x_to_z(p, xlst[-1])
        zlst.append(new_z)

        new_x = z_to_x(p, zlst[-1])
        xlst.append(new_x)
        new_z = x_to_z(p, consider_constant(xlst[-1]))
        zlst.append(new_z)

        new_x = z_to_x(p, zlst[-1])
        xlst.append(new_x)

    else:

        for inds in range(0, num_iterations):
            new_x = z_to_x(p, zlst[-1])
            xlst.append(new_x)
            new_z = x_to_z(p, xlst[-1])
            zlst.append(new_z)

    for j in range(len(xlst)):
        xlst[j] = T.nnet.sigmoid(xlst[j])

    return xlst, zlst


def onestep_z_to_x(p, z):
    x = T.nnet.sigmoid(z_to_x(p, z))
    return x


def onestep_x_to_z(p, x):
    new_z = x_to_z(p, inverse_sigmoid(x))
    return new_z


def q_chain(p, x, num_iterations):

    xlst = [x]
    zlst = []
    new_z = x_to_z(p, inverse_sigmoid(xlst[-1]))
    zlst.append(new_z)

    return xlst, zlst


def make_model():
    logger.info("Initializing parameters")
    gparams = init_gparams({})
    dparams = init_dparams({})

    logger.info("Setting input variables")
    z_in = T.matrix('z_in')
    x_in = T.matrix()

    logger.info("Building graph")
    p_lst_x, p_lst_z = p_chain(gparams, z_in, num_steps)
    q_lst_x, q_lst_z = q_chain(gparams, x_in, num_steps)
    p_lst_x_long, p_lst_z_long = p_chain(gparams, z_in, 19)

    z_inf = q_lst_z[-1]

    logger.debug("p chain x: {}".format(p_lst_x))
    logger.debug("p chain z: {}".format(p_lst_z))
    logger.debug("q chain x: {}".format(q_lst_x))
    logger.debug("q chain z: {}".format(q_lst_z))

    D_p_lst_1, D_feat_p_1 = discriminator(dparams, p_lst_x[-1], p_lst_z[-1])
    D_p_lst_2, D_feat_p_2 = discriminator(dparams, p_lst_x[-2], p_lst_z[-2])

    D_p_lst = D_p_lst_1 + D_p_lst_2

    logger.info("Using double discriminator")

    D_q_lst, D_feat_q = discriminator(dparams, q_lst_x[-1], q_lst_z[-1])

    logger.info("Setting optimizer")
    dloss, gloss = lsgan_loss(D_q_lst, D_p_lst)
    dupdates = lasagne.updates.rmsprop(dloss, dparams.values(), 0.0001)
    gloss_grads = T.grad(gloss, gparams.values(), disconnected_inputs='ignore')
    gupdates = lasagne.updates.rmsprop(gloss_grads, gparams.values(), 0.0001)
    gcupdates = lasagne.updates.rmsprop(gloss, gparams.values(), 0.0001)

    dgupdates = dupdates.copy()
    dgupdates.update(gupdates)

    dgcupdates = dupdates.copy()
    dgcupdates.update(gcupdates)

    logger.info("Compiling functions")
    train_disc_gen_classifier = theano.function(
        inputs=[x_in, z_in], outputs=[dloss, p_lst_x[-1], p_lst_z[-1]],
        updates=dgcupdates, on_unused_input='ignore')

    #get_zinf = theano.function([x_in], outputs=z_inf)
    #get_dfeat = theano.function([x_in], outputs=D_feat_q)

    get_pchain = theano.function([z_in], outputs=p_lst_x_long)
    
    func_z_to_x = theano.function([z_in], outputs=onestep_z_to_x(gparams, z_in))
    func_x_to_z = theano.function([x_in], outputs=onestep_x_to_z(gparams, x_in))


def train():
    z_out_p = rng.normal(size=(batch_size, nl)).astype(floatX)

    for iteration in range(0, n_iterations):

        if persist_p_chain:
            z_in_new = rng.normal(size=(64, nl)).astype(floatX)
            blending = rng.uniform(0.0, 1.0, size=(64,))
            z_in_new[
                blending >= blending_rate] = z_out_p[
                blending >= blending_rate]
            z_in = z_in_new
        else:
            z_in = rng.normal(size=(64, nl)).astype(floatX)

        if latent_sparse:
            z_in[:, 128:] *= 0.0

        r = random.randint(0, 50000 - 64)

        if dataset == "mnist":
            x_in = trainx[r:r + batch_size]

            x_in = x_in.reshape((batch_size, 1, 28, 28))

            x_in = np.repeat(x_in, 3, axis=(1))
            x_in = np.lib.pad(
                x_in, ((0, 0), (0, 0), (2, 2), (2, 2)), 'constant', constant_values=(0))

            x_in = x_in.reshape((64, 32 * 32 * 3))
        elif dataset == "anime":
            x_in = normalize(animeData.getBatch()).reshape((64, 32 * 32 * 3))

        elif dataset == "svhn":
            x_in = normalize(
                svhnData.getBatch()['x']).reshape(
                (64, 32 * 32 * 3))

        dloss, gen_x, z_out_p = train_disc_gen_classifier(x_in, z_in)

        logger.info("Iteration {}".format(iteration))
        logger.info("Discriminator loss: {}".format(dloss))
        logger.info("Generated X mean: {}".format(gen_x.mean()))

        if iteration % updates_to_show == 0:
            logger.info("Discriminator loss: {}".format(dloss))
            plot_images(gen_x, "plots/" + slurm_name + "_gen.png")
            #plot_images(reconstruct(x_in).reshape((64,1,28,28)), "plots/" + slurm_name + "_rec.png")

            # NOT CORRECT INITIALLY
            #rec_loop = [x_in]
            # for b in range(0,9):
            #    rec_loop.append(reconstruct(rec_loop[-1]))
            #    rec_loop[-1][:,0:392] = x_in[:,0:392]
            #    plot_images(rec_loop[-1].reshape((64,1,28,28)), "plots/" + slurm_name + "_rec_" + str(b) +".png")

            plot_images(x_in, "plots/" + slurm_name + "_original.png")

            p_chain = get_pchain(z_in)
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
    
if __name__ == '__main__':
    print "dataset", dataset
    print "num steps", num_steps
    print "train classifier separate", train_classifier_separate
    print "latent sparse", latent_sparse
    print "persistent p chain", persist_p_chain
    print 'blending rate (odds of keeping old z in P chain)', blending_rate

    

