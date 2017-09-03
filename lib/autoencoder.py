
import sys

sys.path.append("/u/lambalex/DeepLearning/undirected_matching")
sys.path.append("/u/lambalex/DeepLearning/undirected_matching/lib")

from nn_layers import param_init_ffcoupling, ffcoupling, init_tparams
import theano
import theano.tensor as T
import numpy.random as rng
import gzip
import cPickle as pickle

mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

train, valid, test = pickle.load(mn)

trainx,trainy = train
validx,validy = valid
testx, testy = test

num_layers = 3
ndim = 784

params = {}
params = param_init_ffcoupling(params, "derp", ndim)
params = param_init_ffcoupling(params, "derp2", ndim)
tparams = init_tparams(params)

x = T.matrix()

z = ffcoupling(tparams, "derp", x, ndim, mode="forward")

xr = ffcoupling(tparams, "derp", z, ndim, mode='reverse')

f = theano.function([x], outputs = xr)

#xin = rng.normal(size=(1,ndim)).astype('float32')

xin = trainx[0:64]

print xin.mean()
print f(xin).mean()

print ((xin - f(xin))**2).mean()


