import numpy as np
import numpy.random as nprand

def binary_rand(n):
    return nprand.randint(0,2,n)

def logistic(a):
    return 1. / (1. + np.exp(-a))

def upsample(rbm, x):
    return logistic(x.dot(rbm.W))

def downsample(rbm, h):
    return logistic(h.dot(rbm.W.T))


class chains:
    def __init__(self, rbm, n_chains):
        self.n_chains  = n_chains
        self.n_visible = rbm.n_visible
        self.n_hidden  = rbm.n_hidden
        self.__dtype = np.dtype('int8')
        self.x = binary_rand((self.n_chains,self.n_visible)).astype(self.__dtype)
        self.update_h(rbm)
        

    #def upsample(self, rbm):
    #    return self.x.dot(rbm.W)

    #def downsample(self, rbm):
    #    return self.h.dot(rbm.W.T)

    def update_x(self, rbm):
        downsampled = downsample(rbm, self.h)
        self.x = (nprand.uniform(0., 1., downsampled.shape) < downsampled).astype(self.__dtype)
        return self.x

    def update_h(self, rbm):
        upsampled = upsample(rbm, self.x)
        self.h = (nprand.uniform(0., 1., upsampled.shape) < upsampled).astype(self.__dtype)
        return self.h

    def alternating_gibbs(self, rbm, n):
        for i in xrange(n):
            self.update_h(rbm)
            self.update_x(rbm)
        return self.x
        

class rbm:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.zeros((n_visible, n_hidden))

    def fit(self, x, n_iterations=100, n_chains = 100, alpha=0.05, lamb=0.05):
        n_instances, n_visible = x.shape
        assert(n_visible == self.n_visible)

        persistent_chains = chains(self, n_chains)
        for i in xrange(n_iterations):
            plus_p = upsample(self, x)
            g_plus = x.T.dot(plus_p)

            sample_x = persistent_chains.alternating_gibbs(self, 1)
            prob_h = upsample(self, sample_x)
            g_minus = sample_x.T.dot(prob_h)

            self.W += alpha * ( (g_plus / n_instances) - 
                                (g_minus / n_chains) - 
                                (lamb * self.W) )
            
            
            
        
        
