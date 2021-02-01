###################################################################
###                                                             ###
###    File composed by Luuk Oudshoorn on behalf of group 18    ###
###             Compatible with standard python3                ###
###      Note: most of the scripts run in parallel. This        ###
###        might cause some problems in case this is not        ###
###           available on the host machine when running.       ###
###            For the RNN part, one needs a working GPU        ###
###                                                             ###
###################################################################
import numpy as np
import pandas as pd
from glob import glob
from scipy.optimize import minimize, approx_fprime
from scipy.special import gamma, factorial, polygamma
from scipy import special
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import matplotlib.pyplot as plt
import scipy
import matplotlib.dates as mdates
import os
import sys
import corner
import time
from multiprocessing import Pool
import datetime
# stuff for mcmc simulations
import emcee
import corner

class emcee_class():
    def __init__(self,daily_returns,fittedGARCHES, fittedRealGARCHES,modeltype='GARCH22',handin=False):
        self.modeltype = modeltype
        self.x = daily_returns.values.flatten()*100
        self.handin = handin
        self.n=len(self.x)
        if modeltype=='GARCH22':
            self.parameter_keys = ['mu','beta1', 'beta2','alpha1','alpha2']
            self.labels = [r'$\omega$',r'$\beta_1$',r'$\beta_2$',r'$\alpha_1$',r'$\alpha_2$']
            self.theta_guess = fittedGARCHES[4].thetahat
        if modeltype=='EGARCH11':
            self.parameter_keys = ['mu','beta1','alpha1','theta']
            self.labels = [r'$\omega$',r'$\beta_1$',r'$\alpha_1$',r'$\theta$']
            self.theta_guess = [-0.236,0.1126,0.9697,-0.8808]
        self.sigmas = np.ones(len(self.parameter_keys))*0.3
        
        
        self.__emcee__()
    
    def model(self,params):
        # Just the same model as in the other fit garch functions
        sigma2 = np.zeros(self.n)
        x = self.x
        # parameters are in a dictionary
        params = dict(zip(self.parameter_keys, params))
        # fill first values with sample variance
        if self.modeltype=='GARCH22':
            sigma2[0] = np.var(x)
            # Iterate through times
            for t in range(1,self.n):
                beta_part = params['beta1']*sigma2[t-1]  + params['beta2']*sigma2[t-2]
                alpha_part = params['alpha1']*(x[t-1]**2) + params['alpha2']*(x[t-2]**2)
                sigma2[t] = params['mu'] + beta_part + alpha_part
            return sigma2
        elif self.modeltype=='EGARCH11':
            sigma2[0] = np.exp(params['mu'] + params['beta1']*np.log(np.std(x)))
            # Iterate through times
            for t in range(1,self.n):
                g = params['theta']*x[t-1] + np.abs(x[t-1])-np.sqrt(2/np.pi)
                sigma2[t] = np.exp(params['mu'] + params['alpha1']*g+params['beta1']*np.log(sigma2[t-1]))
            return sigma2
    

    def lnprior(self,params):
        #P,k,f0,v0 = theta
        sum_ = 0
        for i in range(len(self.parameter_keys)):
            sum_ += (-(params[i] - self.theta_guess[i])**2.)/(2.*self.sigmas[i]**2.)
        return sum_


    def lnL(self,params):
        # Log likelihood; combination of priod and model 
        sigma2 = self.model(params)
        lnp = self.lnprior(params)
        #Prior should be finite
        if not np.isfinite(lnp):
            return -np.infty
        # Sigmas should be positive
        if np.sum(sigma2<0)>0:
            return -np.infty
        params = dict(zip(self.parameter_keys, params))
        # Get likelihood from model
        L = -0.5*np.log(2*np.pi) - 0.5*np.log(sigma2) - 0.5*self.x**2/sigma2
        llik = np.mean(L)
        if np.isfinite(llik):
            return llik + lnp
        else:
            return -np.infty
    
    def __emcee__(self):
        # Set up the properties of the problem.
        ndim= len(self.parameter_keys)
        # Run shallow if handin version
        nwalkers = 2000 if not self.handin else 100
        nsteps   = 2000 if not self.handin else 100
        pos = [self.theta_guess + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
        # Create the sampler.
        from multiprocessing import Pool

        with Pool(20) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnL, pool=pool)
            start = time.time()
            sampler.run_mcmc(pos, nsteps=nsteps, progress=True)
        self.chain = sampler.chain
        
    def __corner__(self):
        # Make corner plot of all parameters
        samples_p = self.chain[:,:, :].reshape((-1, len(self.parameter_keys)))
        fig = plt.figure(figsize=(6,6))
        corner.corner(samples_p, labels=self.labels,label_kwargs={'size':12})
        plt.tight_layout(pad=0.1)
        plt.savefig(self.modeltype+'_corner.pdf',bbox_inches='tight')
        plt.show()

    def __chains__(self):
        # Plot chains
        fig, axes = plt.subplots(ncols=1, nrows=len(self.parameter_keys))
        fig.set_size_inches(12,12)
        # ITerate over axes
        for i in range(len(self.parameter_keys)):
            axes[i].plot(self.chain[:, :, i].transpose(), color='black', alpha=0.3)
            axes[i].axvline(100, ls='dashed', color='red')
        axes[0].axvline(100, ls='dashed', color='red')
        plt.show()
    
    def results(self):
        # Cut of burn in period and get median and errors
        samples = self.chain
        labels = self.labels
        for i in range(len(labels)):
            low,mid,high = np.percentile(samples[:,:,i][500:],q=[16,50,84])
            deltas = np.diff([low,mid,high])
            # Print for easy Latex converting
            print((labels[i][1:-1]+'='+str(np.round(mid,3))+'_{-' 
                  +str(np.round(deltas[0],2)) + '}^{+'+str(np.round(deltas[1],2))+'}'))