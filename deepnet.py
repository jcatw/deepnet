"""
A simple implementation of binary deep belief networks as stacked
restricted boltzmann machines.
"""
import numpy as np
import numpy.random as nprand
import copy
from scipy.optimize import minimize
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
    '''
    This class encapsulates a set of Markov chains running over an RBM layer.
    '''
    def __init__(self, rbm, n_chains):
        self.n_chains  = n_chains
        self.n_visible = rbm.n_visible
        self.n_hidden  = rbm.n_hidden
        self.dtype = rbm.dtype
        self.x = binary_rand((self.n_chains,self.n_visible)).astype(self.dtype)
        self.update_h(rbm)
        
    def update_x(self, rbm):
        '''
        Sample visible variables given hidden.
        '''
        downsampled = downsample(rbm, self.h)
        self.x = (nprand.uniform(0., 1., downsampled.shape) < downsampled).astype(self.dtype)
        return self.x

    def update_h(self, rbm):
        '''
        Sample hidden variables given visible.
        '''
        upsampled = upsample(rbm, self.x)
        self.h = (nprand.uniform(0., 1., upsampled.shape) < upsampled).astype(self.dtype)
        return self.h

    def alternating_gibbs(self, rbm, n):
        '''
        Perform n steps of alternating Gibbs sampling for a given RBM.
        '''
        for i in xrange(n):
            self.update_h(rbm)
            self.update_x(rbm)
        return self.x
        

class rbm:
    def __init__(self, n_visible, n_hidden, dtype = np.dtype('int8')):
        '''
        Initialize a Restricted Boltzmann Machine with n_visible visible variables and n_hidden hidden variables.
        '''
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.zeros((n_visible, n_hidden))
        self.dtype = dtype

    def fit(self, x, n_iterations=100, n_chains = 100, alpha=0.05, lamb=0.05):
        '''
        Fit an RBM given a design matrix x.
        '''
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
        '''Sample n_samples instances from a fit RBM using n_chains parallel
        Markov chains.  burn_in and burn_interval represent the
        initial number of dropped samples and interval of accepted
        samples, respectively.
        '''
        
        assert(n_samples % n_chains == 0)
        markov_chains = chains(self, n_chains)
        samples = np.zeros((n_samples, self.n_visible))
        n_samples_per_chain = n_samples / n_chains
        samples[:n_chains] = markov_chains.alternating_gibbs(self,burn_in)
        for i in xrange(1,n_samples_per_chain):
            samples[i*n_chains:(i+1)*n_chains] = markov_chains.alternating_gibbs(self, burn_interval)
        return samples
            
class dbn:
    def __init__(self,n_visible,n_hidden_list):
        '''
        Initialize a DBN with n_visible hidden variables.  n_hidden_list defines the number of latent layers and their dimensionality.  For instance, to define a DBN with 1000 visible variables and 3 layers with 100 variables each, call dbn(1000, [100,100,100]).
        '''
        
        self.n_layers = len(n_hidden_list) + 1
        self.n_vars = [n_visible]
        self.n_vars.extend(n_hidden_list)
        self.rbms = []
        self.rbms_up = []
        self.rbms_down = []
        self.n_rbms = self.n_layers - 1

    def reset(self):
        '''
        'Unlearn' the dbn by resetting the constituent restricted botzmann machines.
        '''
        self.rbms = []
        self.rbms_up = []
        self.rbms_down = []

    def fit(self, x, epochs = 1, backfit_iterations=100, backfit_rate = 0.001, backfit_gibbs_iterations = 10, n_iterations=100, n_chains = 100, alpha=0.05, lamb=0.05):
        '''
        Fit a DBN via stochastic maximum likelihood followed by backfitting.
        '''
        self.reset()
        n_instances = x.shape[0]
        bottom_data = x

        # fit the restricted boltzmann machines
        for epoch in range(epochs):
            bottom_data = x
            for i in xrange(self.n_layers-1):
                if epoch == 0:
                    an_rbm = rbm(self.n_vars[i],self.n_vars[i+1])
                else:
                    an_rbm = self.rbms[i]
                an_rbm.fit(bottom_data, n_iterations, n_chains, alpha, lamb)
                if epoch == 0:
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

            # topmost state
            up_prob = upsample(self.rbms[-1],bottom_data)
            up_probs.append(up_prob)
            bottom_data = probs_to_binary(up_prob, self.rbms[-1].dtype)
            up_states.append(bottom_data)

            # bottommost state
            up_probs[0] = downsample(self.rbms_down[0],up_states[1])
            
            ## top level
            # this breaks the chain interface a little, could be cleaned up
            #   n_instances chains with no burn-in and burn-interval bi =>
            #   contrastive divergence with bi steps
            top_chains = chains(self.rbms[-1], n_instances)
            # start the chains at the topmost upsampled states
            top_chains.h = up_states[-1]
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
                self.rbms_down[i].W += (backfit_rate * 
                                        up_states[i+1].T.dot((up_states[i] - 
                                                              downsample(self.rbms_down[i],up_states[i+1])))).T
                
                # 'receptive' parameters
                self.rbms_up[i].W += (backfit_rate * 
                                      down_states[i].T.dot((down_states[i+1] - 
                                                            upsample(self.rbms_up[i],down_states[i]))))
            # top level parameters
            self.rbms[-1].W += backfit_rate * (up_states[-2].T.dot(up_states[-1]) - 
                                               down_states[-2].T.dot(down_states[-1]))

    def sample(self, n_samples, n_chains = 1, burn_in = 10, burn_interval = 5):
        '''
        Sample n_samples samples from a fit DBN.
        '''
        layer_samples = self.rbms[-1].sample(n_samples, n_chains, burn_in, burn_interval)
        for i in xrange(self.n_rbms-2,-1,-1):
            layer_samples = probs_to_binary(downsample(self.rbms_down[i], layer_samples), self.rbms_down[i].dtype)
                                                 
        return layer_samples

    def activations(self, x):
        """Compute the top-level activations given an input vector."""

        activs = x
        for i in range(self.n_rbms - 1):
            activs = upsample(self.rbms_up[i], activs)
        activs = upsample(self.rbms[-1], activs)
        return activs

    def optimal_input(self, index):
        if index < 0 or index > self.n_vars[-1] - 1:
            raise IndexError("Index out of range.")
        cons = {'type': 'equality',
                'fun': lambda z: (z**2).sum() - 1.0}
        f = lambda a: self.activations(x)[index]
        x0 = np.zeros(self.n_vars[0])
        x0[0] = 1.0
        optimal_input = minimize(f, x0)
