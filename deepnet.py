"""
A simple implementation of binary deep belief networks as stacked
restricted boltzmann machines.
"""
import numpy as np
import numpy.random as nprand
import copy
import pdb

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

    def sample(self, n_samples, n_chains = 1, burn_in = 10, burn_interval = 5):
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
        self.rbms_up = []
        self.rbms_down = []
        self.n_rbms = self.n_layers - 1

    def fit(self, x, backfit_iterations=100, backfit_rate = 0.001, backfit_gibbs_iterations = 10, n_iterations=100, n_chains = 100, alpha=0.05, lamb=0.05):
        n_instances = x.shape[0]
        bottom_data = x

        # fit the restricted boltzmann machines
        for i in xrange(self.n_layers-1):
            an_rbm = rbm(self.n_vars[i],self.n_vars[i+1])
            an_rbm.fit(bottom_data, n_iterations, n_chains, alpha, lamb)
            self.rbms.append(an_rbm)
            bottom_data = probs_to_binary(upsample(an_rbm, bottom_data), an_rbm.dtype)

        # untie weights and backfit
        ## init untied rbms
        for i in xrange(self.n_rbms - 1):
            self.rbms_up.append(copy.deepcopy(self.rbms[i]))
            self.rbms_down.append(copy.deepcopy(self.rbms[i]))
            
        ## backfit
        for iteration in xrange(backfit_iterations):
            up_states = [x]
            #up_probs = [downsample(self.rbms_down[0],x)]
            up_probs = [None]
            down_states = []
            down_probs = []
            
            ## 'wake'
            bottom_data = x
            for i in xrange(self.n_rbms-1):
                # get prob, state one level up
                up_prob = upsample(self.rbms_up[i],bottom_data)
                up_probs.append(up_prob)
                bottom_data = probs_to_binary(up_prob, self.rbms_up[i].dtype)
                up_states.append(bottom_data)
            
                # copy the rbm while we are here
                # wrong, fix
                #rbm_up = copy.deepcopy(self.rbms[i])
                #self.rbms_up.append(rbm_up)

            up_probs[0] = downsample(self.rbms_down[0],up_states[1])
            
            ## top level
            # this breaks the chain interface a little, could be cleaned up
            #   n_instances chains with no burn-in and burn-interval bi =>
            #   constrastive divergence with bi steps
            top_chains = chains(self.rbms[-1], n_instances)
            top_chains.h = up_states[-1]  # start the chains at the topmost upsampled states
            top_chains.update_x(self.rbms[-1])  # set the penultimate layer
            # alternating-gibbs-sample for n_iterations
            for k in xrange(n_iterations):
                top_chains.update_h(self.rbms[-1])
                top_chains.update_x(self.rbms[-1])
            # record the final states and activations (probabilities)
            down_states.append(top_chains.h)
            down_states.append(top_chains.x)
            down_probs.append(upsample(self.rbms[-1],top_chains.x))
            down_probs.append(downsample(self.rbms[-1],top_chains.h))
            
            ## 'sleep'
            top_data = down_states[1]
            for i in xrange(self.n_rbms-2,-1,-1):
                down_prob = downsample(self.rbms_down[i], top_data)
                down_probs.append(down_prob)
                top_data = probs_to_binary(down_prob, self.rbms_down[i].dtype)
                down_states.append(top_data)
            down_states.reverse()
            down_probs.reverse()

            #pdb.set_trace()
            ## parameter updates
            for i in xrange(self.n_rbms-1):
                # 'generative' parameters
                #self.rbms_down[i].W += backfit_rate * np.outer(up_states[i+1],
                #                                               (up_states[i] - up_probs[i]))
                #self.rbms_down[i].W += (backfit_rate * up_states[i+1].T.dot((up_states[i] - up_probs[i]))).T
                self.rbms_down[i].W += (backfit_rate * 
                                        up_states[i+1].T.dot((up_states[i] - 
                                                              downsample(self.rbms_down[i],up_states[i+1])))).T
                #                                               
                # 'receptive' parameters
                #self.rbms_up[i].W += backfit_rate * np.outer(down_states[i],
                #                                             (down_states[i+1] - down_probs[i+1]))
                #self.rbms_up[i].W += backfit_rate * down_states[i].T.dot((down_states[i+1] - down_probs[i+1]))
                self.rbms_up[i].W += (backfit_rate * 
                                      down_states[i].T.dot((down_states[i+1] - 
                                                            upsample(self.rbms_up[i],down_states[i]))))
            # top level parameters
            #self.rbms[-1].W += backfit_rate * (np.outer(up_states[-2], up_states[-1]) - 
            #                                   np.outer(down_states[-2], down_states[-1]))
            self.rbms[-1].W += backfit_rate * (up_states[-2].T.dot(up_states[-1]) - 
                                               down_states[-2].T.dot(down_states[-1]))

    def sample(self, n_samples, n_chains = 1, burn_in = 10, burn_interval = 5):
        layer_samples = self.rbms[-1].sample(n_samples, n_chains, burn_in, burn_interval)
        for i in xrange(self.n_rbms-2,-1,-1):
            layer_samples = probs_to_binary(downsample(self.rbms_down[i], layer_samples), self.rbms_down[i].dtype)
                                                 
        return layer_samples
            
