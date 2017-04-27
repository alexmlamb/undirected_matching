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

dataset = "mnist"
#dataset = "anime"
#dataset = "svhn"

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

elif dataset == "svhn":

    from load_svhn import SvhnData
    from load_file import normalize, denormalize

    svhnData = SvhnData(mb_size=64,segment="train")

nl = 128
#128 works for nl
nfg = 512
nfd = 512

print "dataset", dataset

#3
num_steps = 3
print "num steps", num_steps

train_classifier_separate = True
print "train classifier separate", train_classifier_separate

#skip_conn = True
#print "skip conn", skip_conn

latent_sparse = False
print "latent sparse", latent_sparse

persist_p_chain = False
print "persistent p chain", persist_p_chain

blending_rate = 0.5
print 'blending rate (odds of keeping old z in P chain)', blending_rate

def init_gparams(p):

    p = param_init_fflayer(options={},params=p,prefix='z_to_x1_1',nin=nl,nout=256*8*4,batch_norm=True)

    p = param_init_convlayer(options={},params=p,prefix='z_to_x1_2',nin=256,nout=128,kernel_len=5,batch_norm=True)

    p = param_init_convlayer(options={},params=p,prefix='z_to_x1_3',nin=128,nout=3,kernel_len=5,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='z_to_x2_1',nin=nl,nout=256*8*4,batch_norm=True)

    p = param_init_convlayer(options={},params=p,prefix='z_to_x2_2',nin=256,nout=128,kernel_len=5,batch_norm=True)

    p = param_init_convlayer(options={},params=p,prefix='z_to_x2_3',nin=128,nout=3,kernel_len=5,batch_norm=False)    


    p = param_init_convlayer(options={},params=p,prefix='x1_to_z_1',nin=3,nout=128,kernel_len=5,batch_norm=True)

    p = param_init_convlayer(options={},params=p,prefix='x1_to_z_2',nin=128,nout=256,kernel_len=5,batch_norm=True)

    p = param_init_fflayer(options={},params=p,prefix='x1_to_z_3',nin=256*8*4,nout=128,kernel_len=5,batch_norm=False)

    p = param_init_convlayer(options={},params=p,prefix='x2_to_z_1',nin=3,nout=128,kernel_len=5,batch_norm=True)

    p = param_init_convlayer(options={},params=p,prefix='x2_to_z_2',nin=128,nout=256,kernel_len=5,batch_norm=True)

    p = param_init_fflayer(options={},params=p,prefix='x2_to_z_3',nin=256*8*4,nout=128,kernel_len=5,batch_norm=False)

    return init_tparams(p)

def init_dparams(p):

    p = param_init_convlayer(options={},params=p,prefix='DC_1',nin=3,nout=128,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='DC_2',nin=128,nout=256,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='DC_3',nin=256,nout=512,kernel_len=5,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='D_1',nin=nl+512*4*4,nout=nfd,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_2',nin=nfd,nout=nfd,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_3',nin=nfd,nout=nfd,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='D_o_1',nin=nfd,nout=1,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_2',nin=nfd,nout=1,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_o_3',nin=nfd,nout=1,ortho=False,batch_norm=False)

    p = param_init_convlayer(options={},params=p,prefix='D_o_4',nin=128,nout=1,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='D_o_5',nin=256,nout=1,kernel_len=5,batch_norm=False)
    p = param_init_convlayer(options={},params=p,prefix='D_o_6',nin=512,nout=1,kernel_len=5,batch_norm=False)

    return init_tparams(p)

def z_to_x1(p,z):

    d1 = fflayer(tparams=p,state_below=z,options={},prefix='z_to_x1_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)')

    d1 = d1.reshape((64,256,8,4))

    d2 = convlayer(tparams=p,state_below=d1,options={},prefix='z_to_x1_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=-2)

    d3 = convlayer(tparams=p,state_below=d2,options={},prefix='z_to_x1_3',activ='lambda x: x',stride=-2)

    return d3

def z_to_x2(p,z):

    d1 = fflayer(tparams=p,state_below=z,options={},prefix='z_to_x2_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)')

    d1 = d1.reshape((64,256,8,4))

    d2 = convlayer(tparams=p,state_below=d1,options={},prefix='z_to_x2_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=-2)

    d3 = convlayer(tparams=p,state_below=d2,options={},prefix='z_to_x2_3',activ='lambda x: x',stride=-2)

    return d3



def x1_to_z(p,x):

    x = x.reshape((64,3,32,32))

    d1 = convlayer(tparams=p,state_below=x,options={},prefix='x1_to_z_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=2)

    d2 = convlayer(tparams=p,state_below=d1,options={},prefix='x1_to_z_2',activ='lambda x: T.nnet.relu(x,alpha=0.02)',stride=2)

    d2 = d2.flatten(2)

    d3 = fflayer(tparams=p,state_below=d2,options={},prefix='x1_to_z_3',activ='lambda x: x')

    return d3


def x2_to_z(p,x):

    x = x.reshape((64,3,32,32))

    d1 = convlayer(tparams=p,state_below=x,options={},prefix='x2_to_z_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=2)

    d2 = convlayer(tparams=p,state_below=d1,options={},prefix='x2_to_z_2',activ='lambda x: T.nnet.relu(x,alpha=0.02)',stride=2)

    d2 = d2.flatten(2)

    d3 = fflayer(tparams=p,state_below=d2,options={},prefix='x2_to_z_3',activ='lambda x: x')

    return d3



def discriminator(p,x,z):

    dc_1 = convlayer(tparams=p,state_below=x.reshape((64,3,32,32)),options={},prefix='DC_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=2)

    dc_2 = convlayer(tparams=p,state_below=dc_1,options={},prefix='DC_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=2)

    dc_3 = convlayer(tparams=p,state_below=dc_2,options={},prefix='DC_3',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',stride=2)

    inp = join2(z,dc_3.flatten(2))

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='D_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',mean_ln=False)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='D_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)', mean_ln=False)

    h3 = fflayer(tparams=p,state_below=h2,options={},prefix='D_3',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)', mean_ln=False)

    D1 = fflayer(tparams=p,state_below=h1,options={},prefix='D_o_1',activ='lambda x: x')
    D2 = fflayer(tparams=p,state_below=h2,options={},prefix='D_o_2',activ='lambda x: x')
    D3 = fflayer(tparams=p,state_below=h3,options={},prefix='D_o_3',activ='lambda x: x')

    D4 = convlayer(tparams=p,state_below=dc_1,options={},prefix='D_o_4',activ='lambda x: x',stride=2)
    D5 = convlayer(tparams=p,state_below=dc_2,options={},prefix='D_o_5',activ='lambda x: x',stride=2)
    D6 = convlayer(tparams=p,state_below=dc_3,options={},prefix='D_o_6',activ='lambda x: x',stride=2)

    print "special thing in D"
    return [D1,D2,D3,D4,D5,D6], h3

def p_chain(p, num_iterations):

    x_initial = srng.normal(size = (64,3,32,32))
    z2_initial = srng.normal(size = (64,128))

    z1_lst = []
    z2_lst = [z2_initial]
    x_lst = [x_initial]

    for inds in range(0,num_iterations):
        z1_new = to_z1(p, x=consider_constant(x_lst[-1]), z2=consider_constant(z2_lst[-1]))
        z1_lst.append(z1_new)

        x_new = to_x(p,z1_lst[-1])
        z2_new = to_z2(p,z1_lst[-1])

        x_lst.append(x_new)
        z2_lst.append(z2_new)

    for j in range(len(x_lst)):
        x_lst[j] = T.nnet.sigmoid(x_lst[j])

    return x_lst, z1_lst, z2_lst

def q_chain(p,x,num_iterations):

    z2_initial = srng.normal(size = (64,128))

    z1_lst = []
    z2_lst = [z2_initial]
    x_lst = [inverse_sigmoid(x)]

    for inds in range(0,num_iterations):
        z1_new = to_z1(p, x=consider_constant(x_lst[-1]), z2=consider_constant(z2_lst[-1]))
        z1_lst.append(z1_new)

        z2_new = to_z2(p,z1_lst[-1])

        z2_lst.append(z2_new)

    return x_lst, z1_lst, z2_lst

gparams = init_gparams({})
dparams = init_dparams({})

#z_in = T.matrix('z_in')
x_in = T.matrix()

p_lst_x,p_lst_z1,p_lst_z2 = p_chain(gparams, num_steps)

q_lst_x,q_lst_z1,q_lst_z2 = q_chain(gparams, x_in, num_steps)

p_lst_x_long,p_lst_z1_long,p_lst_z2_long = p_chain(gparams, 19)


print p_lst_x
print p_lst_z2
print q_lst_x
print q_lst_z2

D_p_lst_1,D_feat_p_1 = discriminator(dparams, p_lst_x[-1], p_lst_z2[-1])
#D_p_lst_2,D_feat_p_2 = discriminator(dparams, p_lst_x[-2], p_lst_z2[-2])

D_p_lst = D_p_lst_1

print "using single disc at end"

D_q_lst,D_feat_q = discriminator(dparams, q_lst_x[-1], q_lst_z2[-1])

dloss, gloss = lsgan_loss(D_q_lst, D_p_lst)

dupdates = lasagne.updates.rmsprop(dloss, dparams.values(),0.0001)
gloss_grads = T.grad(gloss, gparams.values(), disconnected_inputs='ignore')
gupdates = lasagne.updates.rmsprop(gloss_grads, gparams.values(),0.0001)

gcupdates = lasagne.updates.rmsprop(gloss, gparams.values(),0.0001)

dgupdates = dupdates.copy()
dgupdates.update(gupdates)

dgcupdates = dupdates.copy()
dgcupdates.update(gcupdates)

train_disc_gen_classifier = theano.function(inputs = [x_in], outputs=[dloss,p_lst_x[-1],p_lst_z2[-1]], updates=dgcupdates,on_unused_input='ignore')

#get_zinf = theano.function([x_in], outputs=z_inf)
#get_dfeat = theano.function([x_in], outputs=D_feat_q)

get_pchain = theano.function([], outputs = p_lst_x_long)

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
            x_in = trainx[r:r+64]

            x_in = x_in.reshape((64,1,28,28))

            x_in = np.repeat(x_in,3,axis=(1))
            x_in = np.lib.pad(x_in,((0,0),(0,0),(2,2),(2,2)),'constant',constant_values=(0))

            x_in = x_in.reshape((64,32*32*3))
        elif dataset == "anime":
            x_in = normalize(animeData.getBatch()).reshape((64,32*32*3))

        elif dataset == "svhn":
            x_in = normalize(svhnData.getBatch()['x']).reshape((64,32*32*3))

        dloss,gen_x,z_out_p = train_disc_gen_classifier(x_in)

        print "iteration", iteration
        print "dloss", dloss
        print "gen_x mean", gen_x.mean()

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
            
            p_chain = get_pchain()
            for j in range(0,len(p_chain)):
                print "printing element of p_chain", j
                plot_images(p_chain[j], "plots/" + slurm_name + "_pchain_" + str(j) + ".png")




