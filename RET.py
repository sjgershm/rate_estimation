import numpy as np
from scipy.stats import norm
from scipy.special import psi, polygamma
from scipy import integrate

# Rate estimation theory model class constructor
class model_constructor:
    def __init__(self, nStim=2, n0=1, r0=1, beta=1):
        self.n0 = n0                # prior stimulus duration
        self.r0 = r0                # prior number of reinforcements
        self.lambda_hat = np.zeros(nStim) + n0/r0   # initial rate estimates
        self.N = np.zeros(nStim) + n0 # initial stimulus durations
        self.beta = beta            # decision threshold
        
    def predict(self, x):
        # Predict reinforcement
        return np.dot(x,self.lambda_hat)
    
    def CRprob(self, x):
        # Conditioned response probability
        H, V = self.informativeness(x)
        return norm.cdf((H-self.beta)/np.sqrt(V))
    
    def informativeness(self, x):
        # Compute informativeness (CS-US mutual info up to a constant)
        lambda_hat = np.multiply(self.lambda_hat,x[1:])  # select rates for present stimuli
        R_hat = np.dot(self.N[1:],lambda_hat[1:])
        R_hat_b = self.N[0]*self.lambda_hat[0]
        H = psi(R_hat) - psi(R_hat_b) - np.sum(np.log(self.N[1:])) + np.log(self.N[0])
        V = polygamma(1, R_hat) + polygamma(1, R_hat_b)
        return H, V
    
    def run(self, events, t_start, t_end, step_size = 0.5):
        # Run model over a time range. This function breaks the range into a number of 500ms update steps.
        steps = np.arange(t_start, t_end, step_size)
        update = lambda t: self.update(*events(t))
        for i in range(len(steps)-1):
            delta, err = integrate.quad_vec(update, steps[i], steps[i+1], limit=10, quadrature='gk15', epsabs=1e-20, epsrel=1e-05)
            delta_lambda_hat, delta_N = np.split(delta, 2)
            self.lambda_hat += delta_lambda_hat
            self.N += delta_N
    
    def update(self, x, r):
        # Update model
        delta_lambda_hat = np.divide(x,self.N)*(r-self.predict(x))
        delta_N = x
        return np.concatenate([delta_lambda_hat, delta_N])
    

# Events function generator for Pavlovian conditioning protocol with Poisson distributions
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