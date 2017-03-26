'''
-Initially make z_to_x and x_to_z fairly shallow networks.  Inject noise?  

-Use the fflayer class?  

'''
import theano
import theano.tensor as T
from nn_layers import fflayer, param_init_fflayer
from utils import init_tparams, join2, srng
from loss import accuracy, crossent
import lasagne
import numpy as np
import numpy.random as rng
import gzip
import cPickle as pickle
import random
from viz import plot_images

mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

train, valid, test = pickle.load(mn)

trainx,trainy = train
validx,validy = valid
testx, testy = test

m = 784

def init_gparams(p):


    p = param_init_fflayer(options={},params=p,prefix='z_x_1',nin=128,nout=1024,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='z_x_2',nin=1024,nout=1024,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='z_x_3',nin=1024,nout=m,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='x_z_1',nin=m,nout=512,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='x_z_2',nin=512,nout=512,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='x_z_mu',nin=512,nout=128,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='x_z_sigma',nin=512,nout=128,ortho=False,batch_norm=False)

    return init_tparams(p)

def init_dparams(p):


    p = param_init_fflayer(options={},params=p,prefix='D_1',nin=128+m,nout=512,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_2',nin=512,nout=512,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_3',nin=512,nout=1,ortho=False,batch_norm=False)

    return init_tparams(p)

def init_cparams(p):

    p = param_init_fflayer(options={},params=p,prefix='c_1',nin=128,nout=512,ortho=False,batch_norm=False)
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

    h1 = fflayer(tparams=p,state_below=x,options={},prefix='x_z_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False)
    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='x_z_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False)
    sigma = fflayer(tparams=p,state_below=h2,options={},prefix='x_z_mu',activ='lambda x: tensor.exp(x)',batch_norm=False)
    mu = fflayer(tparams=p,state_below=h2,options={},prefix='x_z_sigma',activ='lambda x: x',batch_norm=False)

    eps = srng.normal(size=sigma.shape)

    z = eps*sigma + mu

    return z

def discriminator(p,x,z):
    inp = join2(x,z)

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='D_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False)
    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='D_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=False)
    D = fflayer(tparams=p,state_below=h2,options={},prefix='D_3',activ='lambda x: x',batch_norm=False)

    return D

def p_chain(p, z, num_iterations):
    zlst = [z]
    xlst = []

    xlst.append(z_to_x(p,z))

    for i in range(num_iterations-1):
        zlst.append(x_to_z(p,xlst[-1]))
        xlst.append(z_to_x(p,zlst[-1]))
    
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

p_lst_x,p_lst_z = p_chain(gparams, z_in, 2)

q_lst_x,q_lst_z = q_chain(gparams, x_in)

z_inf = q_lst_z[-1]

print p_lst_x
print p_lst_z
print q_lst_x
print q_lst_z

closs,cacc = classifier(cparams,z_in,true_y)

D_p = discriminator(dparams, p_lst_x[-1], 1.0*p_lst_z[-1])
D_q = discriminator(dparams, q_lst_x[-1], 1.0*q_lst_z[-1])

dloss = T.mean(T.sqr(1.0 - D_q)) + T.mean(T.sqr(0.0 - D_p))
gloss = T.mean(T.sqr(1.0 - D_p)) + T.mean(T.sqr(0.0 - D_q))

cupdates = lasagne.updates.adam(closs, cparams.values(),0.0001,beta1=0.5)

dupdates = lasagne.updates.adam(dloss, dparams.values(),0.0001,beta1=0.5)
gloss_grads = T.grad(gloss, gparams.values(), disconnected_inputs='ignore')
gupdates = lasagne.updates.adam(gloss_grads, gparams.values(),0.0001,beta1=0.5)

train_disc = theano.function(inputs = [x_in,z_in], outputs=[z_inf,dloss],updates=dupdates)
train_gen = theano.function(inputs = [x_in,z_in], outputs=[p_lst_x[-1]],updates=gupdates)
train_classifier = theano.function(inputs = [z_in, true_y], outputs=[cacc], updates=cupdates)


if __name__ == '__main__':

    for iteration in range(0,500000):

        z_in = rng.normal(size=(64,128)).astype('float32')

        r = random.randint(0,50000-64)

        x_in = trainx[r:r+64].reshape((64,784))

        z_inf, dloss = train_disc(x_in,z_in)
        gen_x = train_gen(x_in,z_in)

        acc = train_classifier(z_inf, trainy[r:r+64].astype('int32'))

        if iteration % 1000 == 0:
            print "acc", acc
            print "dloss", dloss
            plot_images(gen_x[0].reshape((64,1,28,28)), "plots/gen.png")



