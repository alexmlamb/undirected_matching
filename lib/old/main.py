#!/usr/bin/env python 

'''
-Initially make z_to_x and x_to_z fairly shallow networks.  Inject noise?  

-Use the fflayer class?  

'''
import sys

sys.path.append("/u/lambalex/DeepLearning/undirected_matching")
sys.path.append("/u/lambalex/DeepLearning/undirected_matching/lib")

import theano
import theano.tensor as T
from nn_layers import fflayer, param_init_fflayer
from utils import init_tparams, join2, srng, dropout, inverse_sigmoid
from loss import accuracy, crossent, lsgan_loss
import lasagne
import numpy as np
import numpy.random as rng
import gzip
import cPickle as pickle
import random
from viz import plot_images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
slurm_name = os.environ["SLURM_JOB_ID"]

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()

dataset = "mnist"
#dataset = "anime"


if dataset == "mnist":
    mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

    train, valid, test = pickle.load(mn)

    trainx,trainy = train
    validx,validy = valid
    testx, testy = test

    m = 784
elif dataset == "anime":
    from load_file import FileData, normalize, denormalize

    loc = "/u/lambalex/DeepLearning/animefaces/datafaces/danbooru-faces/"

    animeData = FileData(loc, 32, 64)

    m = 32*32*3

nl = 128
#128 works for nl
nfg = 1024
nfd = 1024

#3
num_steps = 3
print "num steps", num_steps

train_classifier_separate = True
print "train classifier separate", train_classifier_separate

skip_conn = True
print "skip conn", skip_conn

latent_sparse = False
print "latent sparse", latent_sparse

persist_p_chain = True
print "persistent p chain", persist_p_chain

blending_rate = 0.0
print 'blending rate (odds of keeping old z in P chain)', blending_rate

def init_gparams(p):


    p = param_init_fflayer(options={},params=p,prefix='z_x_1',nin=nl,nout=nfg,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='z_x_2',nin=nfg,nout=nfg,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='z_x_3',nin=nfg,nout=m,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='x_z_1',nin=m,nout=nfg,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='x_z_2',nin=nfg,nout=nfg,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='x_z_mu',nin=nfg,nout=nl,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='x_z_sigma',nin=nfg,nout=nl,ortho=False,batch_norm=False)

    return init_tparams(p)

def init_dparams(p):

    p = param_init_fflayer(options={},params=p,prefix='D_1',nin=m+nl,nout=nfd,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_2',nin=nfd,nout=nfd,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_3',nin=nfd,nout=nfd,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_1',nin=nfd,nout=1,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_2',nin=nfd,nout=1,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_3',nin=nfd,nout=1,ortho=False,batch_norm=False)

    return init_tparams(p)


def z_to_x(p,z):

    inp = z

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='z_x_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)
    

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='z_x_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)


    x = fflayer(tparams=p,state_below=h2,options={},prefix='z_x_3',activ='lambda x: x',batch_norm=False)

    return x

def x_to_z(p,x):

    h1 = fflayer(tparams=p,state_below=x,options={},prefix='x_z_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='x_z_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)
    sigma = fflayer(tparams=p,state_below=h2,options={},prefix='x_z_mu',activ='lambda x: x',batch_norm=False)
    mu = fflayer(tparams=p,state_below=h2,options={},prefix='x_z_sigma',activ='lambda x: x',batch_norm=False)

    eps = srng.normal(size=sigma.shape)

    z = eps*T.nnet.sigmoid(sigma) + mu

    z = (z - T.mean(z, axis=0, keepdims=True)) / (0.001 + T.std(z, axis=0, keepdims=True))

    return z

def discriminator(p,x,z):

    inp = join2(x,z)

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='D_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',mean_ln=False)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='D_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)', mean_ln=False)

    h3 = fflayer(tparams=p,state_below=h2,options={},prefix='D_3',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)', mean_ln=False)

    D1 = fflayer(tparams=p,state_below=h1,options={},prefix='D_o_1',activ='lambda x: x',batch_norm=False)
    D2 = fflayer(tparams=p,state_below=h2,options={},prefix='D_o_2',activ='lambda x: x',batch_norm=False)
    D3 = fflayer(tparams=p,state_below=h3,options={},prefix='D_o_3',activ='lambda x: x',batch_norm=False)

    return [D1,D2,D3], h3

def p_chain(p, z, num_iterations):
    zlst = [z]
    xlst = []

    if num_iterations == 1:
        xlst.append(z_to_x(p,z))

    if num_iterations == 2:
        xlst.append(z_to_x(p,z))
        zlst.append(x_to_z(p,consider_constant(xlst[-1])))
        xlst.append(z_to_x(p,zlst[-1]))

    if num_iterations > 2 and skip_conn:
        print "skip connections"
        eps = 0.9
        print "eps", eps
        #print "BLOCKING GRADIENT THROUGH ALL BUT FINAL LAYER"
        print "NO CONSIDER CONSTANT"

        xlst.append(z_to_x(p,z)*eps)
        for inds in range(0,num_iterations-2):
            zlst.append(x_to_z(p,xlst[-1]))
            xlst.append(z_to_x(p,zlst[-1]) * (eps) + (1-eps)* xlst[-1])
        zlst.append(x_to_z(p,consider_constant(xlst[-1])))
        xlst.append(z_to_x(p,zlst[-1]) * (eps) + (1-eps)*consider_constant(xlst[-1]))


    if num_iterations > 2 and (not skip_conn):
        print "NO skip connections"

        xlst.append(z_to_x(p,z))
        for inds in range(0,num_iterations-2):
            zlst.append(x_to_z(p,xlst[-1]))
            xlst.append(z_to_x(p,zlst[-1]))
        zlst.append(x_to_z(p,consider_constant(xlst[-1])))
        xlst.append(z_to_x(p,zlst[-1]))

    for j in range(0,len(xlst)):
        xlst[j] = T.nnet.sigmoid(xlst[j])



    return xlst, zlst

def q_chain(p,x):

    xlst = [x]
    print "INVERSE SIGMOID IN Q CHAIN"
    #INVERSE SIG TURNED OFF
    zlst = [x_to_z(p,inverse_sigmoid(x))]

    return xlst, zlst

gparams = init_gparams({})
dparams = init_dparams({})

z_in = T.matrix('z_in')
x_in = T.matrix()

p_lst_x,p_lst_z = p_chain(gparams, z_in, num_steps)

q_lst_x,q_lst_z = q_chain(gparams, x_in)

p_lst_x_long,p_lst_z_long = p_chain(gparams, z_in,20)

z_inf = q_lst_z[-1]

print p_lst_x
print p_lst_z
print q_lst_x
print q_lst_z

#TODO: turned off z in discriminator!  Sanity check!  
print "TURNED Z IN DISC ON"
D_p_lst,D_feat_p = discriminator(dparams, p_lst_x[-1], 1.0*p_lst_z[-1])
D_q_lst,D_feat_q = discriminator(dparams, q_lst_x[-1], 1.0*q_lst_z[-1])


dloss, gloss = lsgan_loss(D_q_lst, D_p_lst)



#dloss += 10.0 * T.sqrt(T.sum(T.sqr(T.grad(T.mean(D_q_lst[-1]), q_lst_x[-1]))))
#dloss += 10.0 * T.sqrt(T.sum(T.sqr(T.grad(T.mean(D_p_lst[-1]), p_lst_x[-1]))))
#dloss += 10.0 * T.sqrt(T.sum(T.sqr(T.grad(T.mean(D_q_lst[-1]), q_lst_z[-1]))))
#dloss += 10.0 * T.sqrt(T.sum(T.sqr(T.grad(T.mean(D_p_lst[-1]), p_lst_z[-1]))))

dupdates = lasagne.updates.rmsprop(dloss, dparams.values(),0.0001)
gloss_grads = T.grad(gloss, gparams.values(), disconnected_inputs='ignore')
gupdates = lasagne.updates.rmsprop(gloss_grads, gparams.values(),0.0001)

gcupdates = lasagne.updates.rmsprop(gloss, gparams.values(),0.0001)

dgupdates = dupdates.copy()
dgupdates.update(gupdates)

dgcupdates = dupdates.copy()
dgcupdates.update(gcupdates)

train_disc_gen = theano.function([x_in,z_in],outputs=[dloss,p_lst_x[-1]],updates=dgupdates)

train_disc_gen_classifier = theano.function(inputs = [x_in, z_in], outputs=[dloss,p_lst_x[-1],p_lst_z[-1]], updates=dgcupdates)

get_zinf = theano.function([x_in], outputs=z_inf)
get_dfeat = theano.function([x_in], outputs=D_feat_q)

get_pchain = theano.function([z_in], outputs = p_lst_x_long)

if skip_conn: 
    #reconstruct = theano.function([x_in], outputs = x_in +  z_to_x(gparams,x_to_z(gparams,x_in)))
    x_in_inv = inverse_sigmoid(x_in)
    reconstruct = theano.function([x_in], outputs = T.nnet.sigmoid(x_in_inv +  z_to_x(gparams,x_to_z(gparams,x_in_inv))))

else:
    reconstruct = theano.function([x_in], outputs = z_to_x(gparams,x_to_z(gparams,x_in)))


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

        r = random.randint(0,50000-64)
        
        if dataset == "mnist":
            x_in = trainx[r:r+64].reshape((64,784))
        elif dataset == "anime":
            x_in = normalize(animeData.getBatch()).reshape((64,32*32*3))

        dloss,gen_x,z_out_p = train_disc_gen_classifier(x_in,z_in)

        print "iteration", iteration

        if iteration % 1000 == 0:
            print "dloss", dloss
            plot_images(gen_x, "plots/" + slurm_name + "_gen.png")
            #plot_images(reconstruct(x_in).reshape((64,1,28,28)), "plots/" + slurm_name + "_rec.png")

            #NOT CORRECT INITIALLY
            #rec_loop = [x_in]
            #for b in range(0,9):
            #    rec_loop.append(reconstruct(rec_loop[-1]))
            #    rec_loop[-1][:,0:392] = x_in[:,0:392]
            #    plot_images(rec_loop[-1].reshape((64,1,28,28)), "plots/" + slurm_name + "_rec_" + str(b) +".png")

            plot_images(x_in, "plots/" + slurm_name + "_original.png")
            
            p_chain = get_pchain(z_in)
            for j in range(0,len(p_chain)):
                print "printing element of p_chain", j
                plot_images(p_chain[j], "plots/" + slurm_name + "_pchain_" + str(j) + ".png")




