
import sys

sys.path.append("/u/lambalex/DeepLearning/undirected_matching")
sys.path.append("/u/lambalex/DeepLearning/undirected_matching/lib")

from nn_layers import param_init_ffcoupling, ffcoupling, init_tparams
import theano
import theano.tensor as T
import numpy.random as rng

ndim = 16

params = {}
params = param_init_ffcoupling(params, "derp", ndim)
tparams = init_tparams(params)

x = T.matrix()

z = ffcoupling(tparams, "derp", x, ndim, mode="forward")

xr = ffcoupling(tparams, "derp", z, ndim, mode='reverse')

f = theano.function([x], outputs = xr)

xin = rng.normal(size=(1,ndim)).astype('float32')

print xin
print f(xin)

