'''
-Initially make z_to_x and x_to_z fairly shallow networks.  Inject noise?  

-Use the fflayer class?  

'''
import theano
import theano.tensor as T
from nn_layers import fflayer, param_init_fflayer
from utils init_tparams, join2

m = 1

def gparams(p):


    p = param_init_fflayer(options={},params=p,prefix='z_x_1',nin=128,nout=512,ortho=True,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='z_x_2',nin=512,nout=m,ortho=True,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='x_z_1',nin=m,nout=512,ortho=True,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='x_z_mu',nin=512,nout=128,ortho=True,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='x_z_sigma',nin=512,nout=128,ortho=True,batch_norm=False)

    return init_tparams(p)

def dparams(p):


    p = param_init_fflayer(options={},params=p,prefix='D_1',nin=128+m,nout=512,ortho=True,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_1',nin=512,nout=512,ortho=True,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='D_3',nin=512,nout=1,ortho=True,batch_norm=False)

    return init_tparams(p)

def z_to_x(p,z):
    h1 = fflayer(tparams=p,state_below=z,options={},prefix='z_x_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)
    x = fflayer(tparams=p,state_below=h1,options={},prefix='z_x_2',activ='lambda x: x',batch_norm=False)

    return x

def x_to_z(x):

    h1 = fflayer(tparams=p,state_below=x,options={},prefix='x_z_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)
    sigma = fflayer(tparams=p,state_below=h1,options={},prefix='x_z_mu',activ='lambda x: tensor.exp(x)',batch_norm=False)
    mu = fflayer(tparams=p,state_below=h1,options={},prefix='x_z_sigma',activ='lambda x: x',batch_norm=False)

    eps = srng.normal(size=sigma.shape)

    z = eps*sigma + mu

    return z

def discriminator(p,x,z):
    inp = join(x,z)

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='D_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)
    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='D_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',batch_norm=True)
    D = fflayer(tparams=p,state_below=h2,options={},prefix='D_3',activ='lambda x: x',batch_norm=False)

    return D

def p_chain(z, num_iterations):
    zlst = []
    xlst = []

    xlst.append(z_to_x(z))

    for i in range(num_iterations-1):
        zlst.append(x_to_z(xlst[-1]))
        xlst.append(z_to_x(zlst[-1]))
    
    return xlst, zlst


def q_chain(x):

    xlst = [x]
    zlst = [x_to_z(x)]

    return xlst, zlst

    





