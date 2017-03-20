'''
-Initially make z_to_x and x_to_z fairly shallow networks.  Inject noise?  

-Use the fflayer class?  

'''
import theano
import theano.tensor as T
from nn_layers import fflayer, param_init_fflayer

def gparams(p):

def dparams(p):

def z_to_x(z):

def x_to_z(x):

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

def discriminator(x,z):
    





