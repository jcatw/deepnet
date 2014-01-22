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

def probs_to_binary(probs, dtype):
    return (nprand.uniform(0., 1., probs.shape) < probs).astype(dtype)

class chains:
    def __init__(self, rbm, n_chains):
        self.n_chains  = n_chains
        self.n_visible = rbm.n_visible
        self.n_hidden  = rbm.n_hidden
        self.dtype = rbm.dtype
        self.x = binary_rand((self.n_chains,self.n_visible)).astype(self.dtype)
        self.update_h(rbm)
        

    #def upsample(self, rbm):
    #    return self.x.dot(rbm.W)

    #def downsample(self, rbm):
    #    return self.h.dot(rbm.W.T)

    def update_x(self, rbm):
        downsampled = downsample(rbm, self.h)
        self.x = (nprand.uniform(0., 1., downsampled.shape) < downsampled).astype(self.dtype)
        return self.x

    def update_h(self, rbm):
        upsampled = upsample(rbm, self.x)
        self.h = (nprand.uniform(0., 1., upsampled.shape) < upsampled).astype(self.dtype)
        return self.h

    def alternating_gibbs(self, rbm, n):
        for i in xrange(n):
            self.update_h(rbm)
            self.update_x(rbm)
        return self.x
        

class rbm:
    def __init__(self, n_visible, n_hidden, dtype = np.dtype('int8')):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.zeros((n_visible, n_hidden))
        self.dtype = dtype

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

    def sample(self, n_samples, n_chains, burn_in = 10, burn_interval = 5):
        assert(n_samples % n_chains == 0)
        markov_chains = chains(self, n_chains)
        samples = np.zeros((n_samples, self.n_visible))
        #if not initial_states is None:
        #    if begin_visible:
        #        assert(initial_states.shape == (n_samples, self.n_visible))
        #        markov_chains.x = initial_states
        #    else:
        #        assert(initial_states.shape == (n_samples, self.n_hidden))
        #        markov_chains.h = initial_states
        n_samples_per_chain = n_samples / n_chains
        samples[:n_chains] = markov_chains.alternating_gibbs(self,burn_in)
        for i in xrange(1,n_samples_per_chain):
            samples[i*n_chains:(i+1)*n_chains] = markov_chains.alternating_gibbs(self, burn_interval)
        return samples
            
class dbn:
    def __init__(self,n_visible,n_hidden_list):
        self.n_layers = len(n_hidden_list) + 1
        self.n_vars = [n_visible]
        self.n_vars.extend(n_hidden_list)
        self.rbms = []
        self.n_rbms = self.n_layers - 1

    def fit(self, x, n_iterations=100, n_chains = 100, alpha=0.05, lamb=0.05):
        bottom_data = x
        for i in xrange(self.n_layers-1):
            an_rbm = rbm(self.n_vars[i],self.n_vars[i+1])
            an_rbm.fit(bottom_data, n_iterations, n_chains, alpha, lamb)
            self.rbms.append(an_rbm)
            bottom_data = probs_to_binary(upsample(an_rbm, bottom_data), an_rbm.dtype)

    def sample(self, n_samples, n_chains, burn_in = 10, burn_interval = 5):
        layer_samples = self.rbms[-1].sample(n_samples, n_chains, burn_in, burn_interval)
        for i in xrange(self.n_rbms-2,-1,-1):
            layer_samples = probs_to_binary(downsample(self.rbms[i], layer_samples), self.rbms[i].dtype)
                                                 
        return layer_samples
            
