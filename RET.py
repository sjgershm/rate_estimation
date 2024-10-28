import numpy as np
from scipy.stats import norm
from scipy.special import psi, polygamma
from scipy import integrate

# Rate estimation theory model class constructor
class model_constructor:
    def __init__(self, nStim=2, n0=1, r0=0.1):
        self.n0 = n0                # prior stimulus duration
        self.r0 = r0                # prior number of reinforcements
        self.lambda_hat = np.zeros(nStim) + r0/n0   # initial rate estimates
        self.N = np.zeros(nStim) + n0 # initial stimulus durations
        
    def predict(self, x):
        # Predict reinforcement
        # x - event vector
        return np.dot(x,self.lambda_hat)
    
    def run(self, events, t_start, t_end, step_size = 0.5, eta = 0.7, limit = 2):
        # Run model over a time range. This function breaks the range into a series of small update steps.
        # events - function that takes time as input and returns an event vector (x) and reinforcement (r)
        # t_start - start time for integration
        # t_end - end time for integreation
        # step_size - length of time bin for integration
        # eta - learning rate parameter
        # limit - number of function evaluations per integration call
        steps = np.arange(t_start, t_end, step_size)
        update = lambda t: self.update(*events(t))      # update function to be integrated across time
        for i in range(len(steps)-1):
            self.N += step_size*eta     # update counts
            delta, _ = integrate.quad_vec(update, steps[i], steps[i+1], limit=limit, quadrature='gk15', epsabs=1e-05, epsrel=1e-05)
            self.lambda_hat += delta    # update rate estimates
            self.lambda_hat = np.fmax(0.00000001,self.lambda_hat)   # make sure rate estimates are greater than 0
    
    def update(self, x, r):
        # Update rate estimates
        # x - event vector
        # r - reinforcement
        return np.divide(x,self.N)*(r-self.predict(x))
    
# Events function generator for Pavlovian conditioning protocol with Poisson distributions or standard delay conditioning
def generate_events_function(ISI, ITI, lambda_vector=[]):
    total_trial_time = ISI + ITI

    def events(t):
        
        trial_time = t % total_trial_time
        
        if trial_time < ISI:
            # During stimulus presentation
            stimulus_present = [1]
        else:
            # During intertrial interval
            stimulus_present = [0]
        
        # Add the constant as the first component of the stimulus vector
        x = [1] + stimulus_present
        
        if lambda_vector == []: # delay conditioning
            if (ISI-0.1) < trial_time < ISI:    # allow a small (100ms) window of time for the reinforcement, to accommodate integration error
                reward = 1
            else:
                reward = 0
        else:   # Poisson generator
            # Compute the mean of the Poisson process
            mean_reward = np.dot(x, lambda_vector)
            
            # Sample the reward from the Poisson distribution
            reward = np.random.poisson(mean_reward)
        
        return x, reward

    return events