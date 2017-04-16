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
from nn_layers import fflayer, param_init_fflayer, param_init_convlayer, convlayer
from utils import init_tparams, join2, srng, dropout, inverse_sigmoid, join3
from loss import accuracy, crossent, lsgan_loss, wgan_loss
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

#dataset = "mnist"
dataset = "anime"


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

nl = 512
#128 works for nl
nfg = 512
nfd = 512

#3
num_steps = 3
print "num steps", num_steps

train_classifier_separate = True
print "train classifier separate", train_classifier_separate

skip_conn = True
print "skip conn", skip_conn

latent_sparse = True
print "latent sparse", latent_sparse

persist_p_chain = False
print "persistent p chain", persist_p_chain

blending_rate = 0.5
print 'blending rate (odds of keeping old z in P chain)', blending_rate

def init_gparams(p):

    p = param_init_fflayer(options={},params=p,prefix='z_x_1',nin=nl,nout=512*4*4,ortho=False,batch_norm=False)

    p = param_init_convlayer(options={},params=p,prefix='z_x_2',nin=512,nout=256,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='z_x_3',nin=256,nout=128,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='z_x_4',nin=128,nout=3,kernel_len=5,batch_norm=False)

    p = param_init_convlayer(options={},params=p,prefix='x_z_1',nin=3,nout=64,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='x_z_2',nin=64,nout=128,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='x_z_3',nin=128,nout=256,kernel_len=5,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='x_z_mu',nin=256*4*4,nout=nl,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='x_z_sigma',nin=256*4*4,nout=nl,ortho=False,batch_norm=False)

    return init_tparams(p)

def init_dparams(p):

    p = param_init_convlayer(options={},params=p,prefix='DC_1',nin=3,nout=64,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='DC_2',nin=64,nout=128,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='DC_3',nin=128,nout=256,kernel_len=5,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='D_1',nin=nl+256*4*4,nout=nfd,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_2',nin=nfd,nout=nfd,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_3',nin=nfd,nout=nfd,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_1',nin=nfd,nout=1,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_2',nin=nfd,nout=1,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_3',nin=nfd,nout=1,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='D_o_4',nin=16*16*64,nout=1,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_5',nin=8*8*128,nout=1,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_6',nin=4*4*256,nout=1,ortho=False,batch_norm=False)

    return init_tparams(p)


def z_to_x(p,z):

    inp = z

    h0 = fflayer(tparams=p,state_below=inp,options={},prefix='z_x_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)
    #256 x 6 x 6

    h0 = h0.reshape((64,512,4,4))

    h1 = convlayer(tparams=p,state_below=h0,options={},prefix='z_x_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True,stride=-2)

    h2 = convlayer(tparams=p,state_below=h1,options={},prefix='z_x_3',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True,stride=-2)


    x = convlayer(tparams=p,state_below=h2,options={},prefix='z_x_4',activ='lambda x: x',batch_norm=False,stride=-2)
    #3 x 96 x 96

    #h2 = fflayer(tparams=p,state_below=h1,options={},prefix='z_x_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)


    #x = fflayer(tparams=p,state_below=h2,options={},prefix='z_x_3',activ='lambda x: x',batch_norm=False)

    return x.flatten(2)

def x_to_z(p,x):

    #h1 = fflayer(tparams=p,state_below=x,options={},prefix='x_z_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)
    #h2 = fflayer(tparams=p,state_below=h1,options={},prefix='x_z_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)

    h1 = convlayer(tparams=p,state_below=x.reshape((64,3,32,32)),options={},prefix='x_z_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True,stride=2)
    #48,48

    h2 = convlayer(tparams=p,state_below=h1,options={},prefix='x_z_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True,stride=2)
    #24,24

    h3 = convlayer(tparams=p,state_below=h2,options={},prefix='x_z_3',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True,stride=2)
    #12,12


    ho = h3
    ho = ho.flatten(2)

    sigma = fflayer(tparams=p,state_below=ho,options={},prefix='x_z_mu',activ='lambda x: x',batch_norm=False)
    mu = fflayer(tparams=p,state_below=ho,options={},prefix='x_z_sigma',activ='lambda x: x',batch_norm=False)

    eps = srng.normal(size=sigma.shape)

    z = eps*T.nnet.sigmoid(sigma) + mu

    z = (z - T.mean(z, axis=0, keepdims=True)) / (0.001 + T.std(z, axis=0, keepdims=True))

    return z

def discriminator(p,x,z):

    dc_1 = convlayer(tparams=p,state_below=x.reshape((64,3,32,32)),options={},prefix='DC_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False,stride=2)

    dc_2 = convlayer(tparams=p,state_below=dc_1,options={},prefix='DC_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False,stride=2)

    dc_3 = convlayer(tparams=p,state_below=dc_2,options={},prefix='DC_3',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False,stride=2)

    inp = join2(z,dc_3.flatten(2))

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='D_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',mean_ln=False)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='D_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)', mean_ln=False)

    h3 = fflayer(tparams=p,state_below=h2,options={},prefix='D_3',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)', mean_ln=False)

    D1 = fflayer(tparams=p,state_below=h1,options={},prefix='D_o_1',activ='lambda x: x',batch_norm=False)
    D2 = fflayer(tparams=p,state_below=h2,options={},prefix='D_o_2',activ='lambda x: x',batch_norm=False)
    D3 = fflayer(tparams=p,state_below=h3,options={},prefix='D_o_3',activ='lambda x: x',batch_norm=False)

    D4 = fflayer(tparams=p,state_below=dc_1.flatten(2),options={},prefix='D_o_4',activ='lambda x: x',batch_norm=False)
    D5 = fflayer(tparams=p,state_below=dc_2.flatten(2),options={},prefix='D_o_5',activ='lambda x: x',batch_norm=False)
    D6 = fflayer(tparams=p,state_below=dc_3.flatten(2),options={},prefix='D_o_6',activ='lambda x: x',batch_norm=False)

    return [D1*0.0,D2*0.0,D3*0.0,D4,D5,D6], h3

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




