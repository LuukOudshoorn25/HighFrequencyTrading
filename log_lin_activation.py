from scipy.optimize import minimize
from scipy.optimize import minimize
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

class lin_loglin_realgarch():
    def __init__(self,activation='linear'):# or 'RealGARCH'
        self.options = {'eps':1e-09,
                        'maxiter':2000}
        self.activation = activation
        # store the maxdata parameter for if we do predictions
        
        df = pd.read_hdf('./datafiles/daily_returns.h5')
        
        self.datetimes = df.index
        self.closingreturns = df.values.flatten()*100
        
        self.__RVOL__()
        # Call fit of GARCH
        self.fit_garch(self.closingreturns)
        return
    
    def __RVOL__(self):
        """Obtain realized volatilities from highfreq data"""
        # Convert to logprices
        self.RVOL = pd.read_hdf('./datafiles/RVOL_parallel_GARCH.hdf')
        # Daily returns do not have first day and no weekends, so drop them
        self.RVOL = self.RVOL.dropna().values.flatten()[1:]
        #self.RVOL = RK_values.values.flatten()[1:]
    
    def __llik_fun_GARCH__(self,params,estimate=True):
        x = self.closingreturns
        n = len(x)
        # Convert parameters back from their log normalization
        omega = np.exp(params[0])
        alpha = np.exp(params[1])/(1+np.exp(params[1]))
        beta  = np.exp(params[2])/(1+np.exp(params[2]))

        if self.activation == 'linear':
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(x)
            # Iterate through times
            for t in range(1,n):
                # Obtain beta part (lagged sigma2)
                sigma2[t] = omega + beta*sigma2[t-1] + alpha * self.RVOL[t-1]**2
                # Obtain alpha part (lagged returns)
        elif self.activation =='log-linear':
            sigma2 = np.zeros(n)
            sigma2[0] = np.exp(omega + beta*np.log(np.std(x)))
            # Iterate through times
            for t in range(1,n):
                # Obtain beta part (lagged sigma2)
                sigma2[t] = np.exp(omega + beta*np.log(sigma2[t-1]) + alpha * np.log(self.RVOL[t-1]**2))
                # Obtain alpha part (lagged returns)
            
        # Derive likelihood
        if estimate:
            L = -0.5*np.log(2*np.pi) - 0.5*np.log(sigma2) - 0.5*x**2/sigma2

            llik = np.mean(L)

            return -1*llik
        else:
            return sigma2
    
    
    def fit_garch(self,x):
        # Initialize values
        a,b = 0.1,0.4
        alpha = np.log(a/(1-a))
        beta = np.log(b/(1-b))
        omega = np.nanvar(self.closingreturns)*(1-a-b)
        
        par_ini = np.array([omega,alpha,beta])
        
        est = minimize(self.__llik_fun_GARCH__, x0=par_ini,
                       options = self.options,
                       method = 'L-BFGS-B'
                       #bounds = ((0.0001, 100), (0, 10), (0,1))
                      )
        llikhood = -est.fun
        self.AIC = 2*(len(par_ini))-2*llikhood
        self.BIC = (len(par_ini)) * np.log(len(self.closingreturns)) - 2*llikhood
        self.llik_opt = llikhood
        
        self.estimates = est.x
        
        omega_hat = np.exp(self.estimates[0])
        alpha_hat = np.exp(self.estimates[1])/(1+np.exp(self.estimates[1]))
        beta_hat  = np.exp(self.estimates[2])/(1+np.exp(self.estimates[2]))
        
        self.thetahat = np.array([omega_hat, alpha_hat,beta_hat])
        
        
    def return_vola(self):
        sigma2 = self.__llik_fun_GARCH__(self.estimates,estimate=False)
        return self.datetimes,sigma2
    
    def return_llik_AIC_BIC(self):
        return self.AIC,self.llik_opt



def plot_GARCHestimates(m_lin, m_loglin):
    RGARCHlin_x,RGARCHlin_y = m_lin.return_vola()
    RGARCHlog_x,RGARCHlog_y = m_loglin.return_vola()
    
    fig,[ax1,ax2]=plt.subplots(nrows=2)
    ax1.plot(RGARCHlin_x,np.sqrt(252*RGARCHlin_y),label='Linear(1,1)',lw=0.8,color='dodgerblue')
    ax1.plot(RGARCHlog_x,np.sqrt(252*RGARCHlog_y),label='Log-Linear(1,1)',lw=0.8,color='tomato')
    
    ratio = np.sqrt(252*RGARCHlin_y)/np.sqrt(252*RGARCHlog_y)
    meanratio = np.mean(ratio)
    print('Average ratio of realgarch over garch',meanratio)
    
    ax2.plot(RGARCHlog_x,ratio,label='Ratio',lw=0.8,color='grey')
    #ax2.plot(RGARCH33x,np.sqrt(252*RGARCH33y),label='RealGARCH(3,3)',lw=0.8,color='tomato')
    ax1.set_xlim([datetime.date(2018, 6, 1), datetime.date(2020, 12, 1)])
    ax1.set_xticklabels(['']*len(ax1.get_xticks()))
    ax2.set_xlim([datetime.date(2018, 6, 1), datetime.date(2020, 12, 1)])
    

    plt.xlabel('Date')
    plt.xticks(rotation=45)
    ax1.set_ylabel('Volatility (% p.a.)')
    ax2.set_ylabel(r'$\frac{RealGARCH(1,1)}{GARCH(1,1)}$')
    ax1.legend(frameon=True,loc='upper left')
    ax2.legend(frameon=True,loc='upper left')
    plt.tight_layout(pad=0.1)
    ax1.set_ylim(0,180)
    ax2.set_ylim(0,2.9)
    
    ax2.axhline(1,lw=0.4,ls='--',color='black')
    #plt.savefig('Garch_RealGARCH_RVOL.pdf',bbox_inches='tight')
    plt.show()
