import theano
import theano.tensor as T

srng = theano.tensor.shared_randomstreams.RandomStreams(42)

import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy.random as rng

class StochasticBernoulli(theano.Op):
    """
    
    Given a matrix of probabilities, sample from bernoullis.  Then when backpropping, pretend as if the 
    sampling action were the identity transformation.  


    """
    def make_node(self, *inputs):
        logits, noise = inputs
        #logits = T.as_tensor_variable(logits)
        #noise = T.as_tensor_variable(noise)
        return theano.Apply(self, [logits, noise], [logits.type()])

    def perform(self, node, inputs, output_storage):
        logits, noise = inputs
        #sampled_indices = np.argmax(logits + -np.log(-np.log(noise)), axis=-1)
        #one_hot_vectors = np.eye(logits.shape[-1], dtype=np.float32)[sampled_indices]
        
        diff = np.less(noise,logits).astype('float32')

        #diff = (diff - T.sqr(diff))/T.sqr(diff)

        output_storage[0][0] = diff

    def grad(self, inputs, grads):
        logits, noise = inputs
        grad, = grads

        #logits_2d = logits.reshape((-1, logits.shape[-1]))
        #grad_2d = grad.reshape((-1, logits.shape[-1]))

        #softmax_output_2d = T.nnet.softmax(logits_2d)
        #grad_wrt_logits_2d = softmax_output_2d#T.nnet.softmax_grad(grad_2d, softmax_output_2d)

        #grad_wrt_logits = grad_wrt_logits_2d.reshape(logits.shape)
        #error_comment = 'Gradient with respect to noise is not required for backprop, so it is not implemented.'
        #grad_wrt_noise = T.grad_not_implemented(self, 1, noise, comment=error_comment)

        grad_wrt_logits = grad
        grad_wrt_noise = grad

        return [grad_wrt_logits, grad_wrt_noise]


def stochastic_bernoulli(logits):
    random_streams = RandomStreams()
    noise = random_streams.uniform(logits.shape)
    return StochasticBernoulli()(logits, noise)


y = T.matrix()

#p = T.nnet.sigmoid(y)

#s = srng.binomial(n=1,p=p,size=p.shape)


s = stochastic_bernoulli(y)

f = theano.function(inputs = [y], outputs = [s, T.grad(((s - 0.5)**2).sum(),y)])

#x = rng.uniform(size=(10,10)).astype('float32')

x = np.array([[0.0,0.0,0.0,0.0,0.9,0.91,0.9]]).astype('float32')

print x
r = f(x)

print r[0]
print "grad", r[1]


