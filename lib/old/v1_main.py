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
from utils import init_tparams, join2, srng, dropout
from loss import accuracy, crossent
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

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()


mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

train, valid, test = pickle.load(mn)

trainx,trainy = train
validx,validy = valid
testx, testy = test

m = 784
nl = 128
nfg = 1024
nfd = 1024

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

def init_cparams(p):

    p = param_init_fflayer(options={},params=p,prefix='c_1',nin=nl,nout=512,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='c_2',nin=512,nout=10,ortho=False,batch_norm=False)

    return init_tparams(p)

def classifier(p,z,true_y):

    h1 = fflayer(tparams=p,state_below=z,options={},prefix='c_1',activ='lambda x: tensor.nnet.relu(x)',batch_norm=False)
    y_est = fflayer(tparams=p,state_below=h1,options={},prefix='c_2',activ='lambda x: x',batch_norm=False)

    y_est = T.nnet.softmax(y_est)

    acc = accuracy(y_est,true_y)
    loss = crossent(y_est,true_y)

    return loss,acc

def z_to_x(p,z):


    h1 = fflayer(tparams=p,state_below=z,options={},prefix='z_x_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)
    

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='z_x_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)


    x = fflayer(tparams=p,state_below=h2,options={},prefix='z_x_3',activ='lambda x: tensor.nnet.sigmoid(x)',batch_norm=False)

    return x

def x_to_z(p,x):

    h1 = fflayer(tparams=p,state_below=x,options={},prefix='x_z_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='x_z_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)
    sigma = fflayer(tparams=p,state_below=h2,options={},prefix='x_z_mu',activ='lambda x: x',batch_norm=False)
    mu = fflayer(tparams=p,state_below=h2,options={},prefix='x_z_sigma',activ='lambda x: x',batch_norm=False)

    eps = srng.normal(size=sigma.shape)

    z = eps*T.nnet.sigmoid(sigma)*1.0 + mu

    z = (z - T.mean(z, axis=0, keepdims=True)) / (0.001 + T.std(z, axis=0, keepdims=True))

    return z

def discriminator(p,x,z):

    #hx = fflayer(tparams=p,state_below=x,options={},prefix='D_x',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',mean_ln=False)
    #hz = fflayer(tparams=p,state_below=z,options={},prefix='D_z',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',mean_ln=False)

    inp = join2(x,z)

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='D_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',mean_ln=False)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='D_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)', mean_ln=False)

    h3 = fflayer(tparams=p,state_below=h1,options={},prefix='D_3',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)', mean_ln=False)

    D1 = fflayer(tparams=p,state_below=h1,options={},prefix='D_o_1',activ='lambda x: x',batch_norm=False)
    D2 = fflayer(tparams=p,state_below=h2,options={},prefix='D_o_2',activ='lambda x: x',batch_norm=False)
    D3 = fflayer(tparams=p,state_below=h3,options={},prefix='D_o_3',activ='lambda x: x',batch_norm=False)

    return [D1,D2,D3]

def p_chain(p, z, num_iterations):
    zlst = [z]
    xlst = []

    if num_iterations == 1:
        xlst.append(z_to_x(p,z))

    if num_iterations == 2:
        xlst.append(z_to_x(p,z))
        zlst.append(x_to_z(p,xlst[-1]))
        xlst.append(z_to_x(p,zlst[-1]))

    if num_iterations == 3:
        xlst.append(z_to_x(p,z))
        zlst.append(x_to_z(p,xlst[-1]))
        xlst.append(z_to_x(p,zlst[-1]))
        zlst.append(x_to_z(p,consider_constant(xlst[-1])))
        xlst.append(z_to_x(p,zlst[-1]))


    #xlst.append(z_to_x(p,z))

    #for i in range(num_iterations-1):
    #    zlst.append(x_to_z(p,xlst[-1]))
    #    xlst.append(z_to_x(p,zlst[-1]))

    return xlst, zlst


def q_chain(p,x):

    xlst = [x]
    zlst = [x_to_z(p,x)]

    return xlst, zlst

gparams = init_gparams({})
dparams = init_dparams({})
cparams = init_cparams({})

z_in = T.matrix()
x_in = T.matrix()
true_y = T.ivector()

p_lst_x,p_lst_z = p_chain(gparams, z_in, 3)

q_lst_x,q_lst_z = q_chain(gparams, x_in)

z_inf = q_lst_z[-1]

print p_lst_x
print p_lst_z
print q_lst_x
print q_lst_z

closs,cacc = classifier(cparams,z_in,true_y)

D_p_lst = discriminator(dparams, p_lst_x[-1], p_lst_z[-1])
D_q_lst = discriminator(dparams, q_lst_x[-1], q_lst_z[-1])

def lsgan_loss(D_q_lst, D_p_lst):
    dloss = 0.0
    gloss = 0.0

    for i in range(len(D_q_lst)):
        D_q = D_q_lst[i]
        D_p = D_p_lst[i]
        dloss += T.mean(T.sqr(1.0 - D_q)) + T.mean(T.sqr(0.0 - D_p))
        gloss += T.mean(T.sqr(1.0 - D_p)) + T.mean(T.sqr(0.0 - D_q))

    return dloss / len(D_q_lst), gloss / len(D_q_lst)

dloss, gloss = lsgan_loss(D_q_lst, D_p_lst)

cupdates = lasagne.updates.rmsprop(closs, cparams.values(),0.0001)

dupdates = lasagne.updates.rmsprop(dloss, dparams.values(),0.0001)
gloss_grads = T.grad(gloss, gparams.values(), disconnected_inputs='ignore')
gupdates = lasagne.updates.rmsprop(gloss_grads, gparams.values(),0.0001)

#train_disc = theano.function(inputs = [x_in,z_in], outputs=[z_inf,dloss],updates=dupdates)
#train_gen = theano.function(inputs = [x_in,z_in], outputs=[p_lst_x[-1]],updates=gupdates)

dgupdates = dupdates
dgupdates.update(gupdates)

train_disc_gen = theano.function([x_in,z_in],outputs=[z_inf,dloss,p_lst_x[-1]],updates=dgupdates)
train_classifier = theano.function(inputs = [z_in, true_y], outputs=[cacc], updates=cupdates)
get_zinf = theano.function([x_in], outputs=z_inf)

reconstruct = theano.function([x_in], outputs = z_to_x(gparams,x_to_z(gparams,x_in)))

if __name__ == '__main__':

    for iteration in range(0,500000):

        z_in = rng.normal(size=(64,nl)).astype('float32')

        r = random.randint(0,50000-64)

        x_in = trainx[r:r+64].reshape((64,784))

        z_inf, dloss,gen_x = train_disc_gen(x_in,z_in)

        acc = train_classifier(z_inf, trainy[r:r+64].astype('int32'))

        if iteration % 1000 == 0:
            print iteration, "acc", acc
            print "dloss", dloss
            plot_images(gen_x.reshape((64,1,28,28)), "plots/gen.png")
            plot_images(reconstruct(x_in).reshape((64,1,28,28)), "plots/rec.png")
            plot_images(x_in.reshape((64,1,28,28)), "plots/original.png")
            
            z_inf = get_zinf(trainx[0:200])
            print "z_inf shape", z_inf.shape
            plt.scatter(z_inf[:,0], z_inf[:,1],c=trainy[0:200])
            plt.savefig("plots/zinf.png")
            plt.clf()




