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


class estimate_GAS():
    def __init__(self,model, maxdate=None):
        
        self.options = {'eps':1e-08,
                      'maxiter':2000}
        self.model = model # t-GAS
        
        self.maxdate = maxdate
        if not maxdate:
            df = pd.read_hdf('daily_returns.h5')
            all_returns = df.copy()
            self.datetimes = df.index
            self.closingreturns = df.values.flatten()
        else: 
            df = pd.read_hdf('daily_returns.h5')
            all_returns = df.copy()
            df = df[:self.maxdate]
            self.datetimes = df.index
            self.closingreturns = df.values.flatten()
        self.closingreturns *=100    
        self.all_returns = all_returns.values.flatten()*100

        # Call fit of GAS
        self.fit_gas(self.closingreturns)
        return
    
    def initialize(self, params):
        h = np.exp(params['omega']+params['alpha']*np.log(np.std(self.all_returns)))
        return h

    def nabla(self,lagged_obs,lagged_latent,params):
        ht = np.exp(lagged_latent)
        
        # Normal distribution
        if self.model == 'G-GAS':
            return -1/2 + lagged_obs**2/(2*ht)
        # t-distribution density
        if self.model == 't-GAS':
            beta = params['beta']
            c=-0.5
            a=(beta+1)*lagged_obs**2
            b=2*(lagged_obs**2+(beta-2)*ht)
            return c+a/b
        if self.model == 'GED':
            beta = params['beta']
            c=-0.5
            lambda_ = np.sqrt(gamma(1/beta)/(2**(2/beta)*gamma(3/beta)))
            a=lagged_obs**2*beta * np.abs(lagged_obs/(lambda_*np.sqrt(ht)))**(beta-2)
            b = 4*lambda_**2*ht
            return c+a/b
        if self.model == 'skewed_t':
            beta = params['beta']
            c=-0.5
            xhi = params['xhi']
            # Restriction: Beta > 2
            m = (gamma((beta-1)/2))/(gamma(beta/2))*(np.sqrt((beta-2)/np.pi))*(xhi-(1/xhi))
            s = np.sqrt(xhi**2+(1/xhi**2)-1-m**2)
            rho = s*((lagged_obs)/np.sqrt(ht)) + m
            if rho >= 0:
                It = 1
            elif rho < 0:
                It = -1
            else:
                It = np.nan
            a = lagged_obs*s*(beta+1)*rho*xhi**(-2*It)
            b = 2*(beta-2)*np.sqrt(ht)*(1+(rho**2*(xhi**(-2*It)))/(beta-2))
            return c+a/b        
        
    def __llik_fun_GAS__(self,params,estimate=True):
        x = self.closingreturns
        n = len(x)
        sigma2 = np.zeros(n)

        exp_abs = np.mean(np.abs(self.all_returns**2))

        params = dict(zip(self.parameter_keys, params))

        # fill first value with sample variance
        sigma2[0] = self.initialize(params)

        # Iterate through times
        for t in range(1,n):
            beta_part = params['phi']*np.log(sigma2[t-1])
            alpha_part = params['alpha']*(np.abs(x[t-1])-exp_abs)+params['gamma']*self.nabla(x[t-1],np.log(sigma2[t-1]), params)
            sigma2[t] = np.exp(params['omega'] + beta_part + alpha_part)
        # Derive likelihood
        if estimate:
            # normal
            if self.model == 'G-GAS':
                L = -0.5*np.log(2*np.pi) - 0.5*np.log(sigma2) - 0.5*self.closingreturns**2/sigma2
                
            # t-distribution
            elif self.model == 't-GAS':
                L = (np.log(gamma((params['beta']+1)/2)) - np.log(gamma(params['beta']/2)) 
                     - 0.5*np.log((params['beta']-2)*np.pi*sigma2)
                     - ((params['beta']+1)/2)*np.log(1+((x**2)/((params['beta']-2)*sigma2))))
                
            # skewed-t-distribution
            elif self.model == 'skewed_t':
                m = ((gamma((params['beta']-1)/2))/(gamma(params['beta']/2)))*(np.sqrt((params['beta']-2)/np.pi))*(params['xhi']-(1/params['xhi']))
                s = np.sqrt(params['xhi']**2+(1/params['xhi']**2)-1-m**2)
                rho = s*((x)/np.sqrt(sigma2)) + m
                pos = (rho>0).astype(float)
                neg = (rho<0).astype(float)
                It = pos-neg
                L = (np.log(gamma((params['beta']+1)/2))-np.log(gamma(params['beta']/2))
                     - 0.5*np.log((params['beta']-2)*np.pi*sigma2)
                     + np.log(s) + np.log((2/(params['xhi']+(1/params['xhi']))))
                     - ((params['beta']+1)/2)*np.log(1+(rho**2)*(params['xhi']**(-2*It))/(params['beta']-2)))
            
            # generalized error distribution
            elif self.model == 'GED':
                lambda_ = np.sqrt(gamma(1/params['beta'])/(2**(2/params['beta'])*gamma(3/params['beta'])))
                L = (-np.log(2**(1+(1/params['beta']))*gamma(1/params['beta'])*lambda_) 
                     - 0.5*np.log(sigma2) + np.log(params['beta']) 
                     - 0.5 * np.abs(x/(lambda_*np.sqrt(sigma2)))**(params['beta']))
                
            llik = np.mean(L)

            return -1*llik
        else:
            return sigma2
    
    
    def fit_gas(self,x):
        # Initialize values
        
        if self.model == 'G-GAS':
            self.parameter_keys = (['phi'] + ['omega'] + ['gamma'] 
                               + ['alpha'])
            par_ini = [ 0.75755154,  0.66283244, -0.00751593,  0.12505432]
            bounds_model = [(0,1)] + (len(par_ini)-1)*[(-np.infty,np.infty)] 
            method_model = 'SLSQP'
            
        if self.model == 't-GAS':
            self.parameter_keys = (['phi'] + ['omega'] 
                                + ['gamma']
                                    + ['alpha']
                                + ['beta'] )
            par_ini = [0.84303148, -1.18629176,  0.44884131,  0.08459087,  3.1]       
            bounds_model = [(0,1)] + (len(par_ini)-2)*[(-np.infty,np.infty)] + [(2.9,3.1)]
            method_model = 'L-BFGS-B'
            
        if self.model == 'skewed_t':
            self.parameter_keys = (['phi'] + ['omega'] 
                                + ['gamma']
                                    + ['alpha']
                                + ['beta'] + ['xhi'])
            par_ini = [0.85669981, 0.29208775, 0.21810941, 0.03660219, 5.90207514, 0.85614126]
            bounds_model = [(-1,1)] + (len(par_ini)-3)*[(-np.infty,np.infty)] + [(2,np.infty)] + [(0.1,np.infty)]   
            method_model = 'L-BFGS-B'
        
        if self.model == 'GED':
            self.parameter_keys = (['phi'] + ['omega'] 
                                + ['gamma']
                                    + ['alpha']
                                + ['beta'] )
            par_ini = [0.72865177, 0.63000793, 0.34841718, 0.4309946 , 0.38422541]
            bounds_model = [(0,1)] + (len(par_ini)-1)*[(-np.infty,np.infty)]
            method_model = 'SLSQP'
        
        
        Lprime = lambda x: approx_fprime(x, self.__llik_fun_GAS__, 0.01)  
        est = minimize(self.__llik_fun_GAS__, x0=par_ini,
                       options = self.options,
                       method = method_model,
                       jac = Lprime,
                       bounds = bounds_model
                      )
        self.estimates = est.x
        
    def __one_step__(self):
        # Obtain historical sigmas (on which the model was trained)
        # and obtain past returns
        _,conditional_sigma = self.return_vola()
        x = self.closingreturns
        # Get fitted parameters
        params = dict(zip(self.parameter_keys, self.estimates))
        
        exp_abs = np.mean(np.abs(self.all_returns**2))

        t = len(conditional_sigma)
        beta_part = params['phi']*np.log(conditional_sigma[t-1])
        
        alpha_part   = params['alpha']*(np.abs(x[t-1])-exp_abs)+params['gamma']*self.nabla(x[t-1],np.log(conditional_sigma[t-1]), params)
        sigma_future = np.exp(params['mu'] + beta_part + alpha_part)
        return sigma_future
    
    
    def return_vola(self,annual=False):
        sigma2 = self.__llik_fun_GAS__(self.estimates,estimate=False)
        sigma2 = sigma2 * 6.5/24
        if not annual:
            return self.datetimes,sigma2
        else:
            return self.datetimes,np.sqrt(252*sigma2)




# Now predict one step ahead
class GAS_RealGAS_predict():
    def __init__(self,RK_values):
        # Create an array of future dates. We iterate over these days, fit a model and predict the first 
        # next dates volatility
        self.RK_values = RK_values
        self.get_dates()
        
        self.models = ['t-GAS','G-GAS','skewed_t']
        
    def simultaneous_fit(self,end_day=None):
        N = np.where(np.array(self.end_days) == end_day)[0][0]
        eta = ((time.time() - self.starttime)/(N+1))*(len(self.end_days)-N)
        print('Fittig models with dates up to',end_day, 'ETA',np.round(eta,1), ' sec')
        # Get list of all modelsup to (3,3)
        
        # Fit all models with data up to the end date passed in the function
        models = self.models
        print(end_day, len(end_day))
        estimated_models = [estimate_GAS(w,maxdate=end_day) for w in models]
        # Obtain point estimate for future sigma2
        point_ests = np.zeros(len(estimated_models))
        for i,model in enumerate(estimated_models):
            point_ests[i] = model.__one_step__()
        return point_ests
        
    def get_dates(self):
        self.end_days = [w.strftime('%Y-%m-%d') for w in pd.to_datetime(self.RK_values['2019':].index)]
        
        
    def sim_fit(self):
        self.starttime = time.time()
        predictions = Parallel(n_jobs=20)(delayed(self.simultaneous_fit)(day) for day in self.end_days)
        self.predictions = np.array(predictions)
        return self.predictions

    def evaluate(self):
        # compare predictions with Realized Kernel volatilities
        # Make dataframe to store results
        results_df = pd.DataFrame({'True',})
        acronyms = ['Pred_'+w[0]+('_R' if w[1] else '') for w in self.models]
        acronyms = dict(zip(acronyms,[np.array([])]*len(acronyms)))
        acronyms['True']=np.array([])
        acronyms['Date']=np.array([])
        df = pd.DataFrame.from_dict(acronyms)
        df = df.set_index('Date')
        for i in range(len(self.end_days)-1):
            date = pd.Series(self.RK_values['2019':].index).iloc[i+1].strftime('%Y-%m-%d')
            true = self.RK_values['2019':].iloc[i+1].values
            prediction = predictions[i]
            df.loc[date] = np.hstack((prediction,true))
        df.index = pd.to_datetime(df.index)
        return df
    
    def get_scores(self):
        # Collect predictions and true
        df = self.evaluate()
        # Drop Covid month
        todrop = df['2020-02-15':'2020-04-15'].index
        df = df.drop(todrop)
        # Convert to annual volas
        df = np.sqrt(252*df)
        # Convert predictions to intraday volas
        for col in df.columns:
            if not col=='True':
                df[col] = df[col] / np.sqrt(24/6.5)
        # Create output df
        score_df = pd.DataFrame({'Model':[],'MAE':[],'RMSE':[]}).set_index('Model')
        # Obtain MAE/RMSE
        for col in df.columns:
            if not col=='True':
                mae = mean_absolute_error(df['True'],df[col])
                rmse = np.sqrt(mean_squared_error(df['True'],df[col]))
                score_df.loc[col] = [mae,rmse]
        return score_df



def real_tGAS(daily_returns,RK_values):
    # Student t-RGAS
    r = daily_returns.values.flatten()*100
    X = RK_values.values.flatten()[1:]#*100

    n = len(X)

    parkeys = ['omega','beta','alpha','nu','nu1']
    par_ini = np.array([0.0129, 0.9296, 0.0657, 6.1143, 6.7575])


    def model(params,annual=False):
        f = np.zeros(n)
        f[0] = np.exp(params['omega']+params['alpha']*np.log(np.std(r)))
        for t in range(0,n-1):
            q1 = params['nu']/2 * (X[t]/np.exp(f[t])-1)
            q2 = -0.5 + ((params['nu1']+1)/2)*(r[t]**2)/((params['nu1']-2)*np.exp(f[t])+r[t]**2)
            q = q1+q2
            f[t+1]  = params['omega'] +params['beta']*f[t]+params['alpha']*q
        if not annual:
            return f
        else:
            sigma = np.sqrt(252*np.exp(f))
            return sigma[sigma>0]

    def llikhood(parvals):
        params = dict(zip(parkeys,parvals))

        f = model(params)
        L1 = -np.log(gamma(params['nu']/2))-(params['nu']/2)*np.log(2*np.exp(f)/params['nu'])
        L2 = (params['nu']/2-1)*np.log(X)-(params['nu']*X/(2*np.exp(f)))
        L3 = np.log(gamma((params['nu1']+1)/2)) - np.log(gamma(params['nu1']/2)) - 0.5*np.log((params['nu1']-2)*np.pi*np.exp(f))
        L4 = -(params['nu1']+1)/2 * np.log(1+r**2 /((params['nu1']-2)*np.exp(f)))
        llikhood = np.nanmean(L1+L2+L3+L4)
        if not np.isfinite(llikhood):
            return np.inf
        return -llikhood

    def optim():
        Lprime = lambda x: approx_fprime(x, llikhood, 0.01)  

        est = minimize(llikhood, x0=par_ini,
                       options = {'eps':1e-10,'maxiter':2000},
                       method = 'Newton-CG',jac=Lprime,
                       bounds = [(0,10),(0,10),(0,10),(0,10),(2,10)])
        estimates = est.x
        print(est.fun)
        print(estimates)
        return estimates


    est = optim()
    est = dict(zip(parkeys,est))
    plt.plot(model(est,annual=True),lw=1)
    return model(est,annual=True)


def real_GGAS(daily_returns,RK_values):
    # G-GAS
    r = daily_returns.values.flatten()*100
    X = RK_values.values.flatten()[1:]#*100

    n = len(X)

    parkeys = ['omega','beta','alpha','nu']
    par_ini = np.array([0.0129, 0.9296, 0.0657, 6.1143])


    def model(params,annual=False):
        f = np.zeros(n)
        f[0] = np.exp(params['omega']+params['alpha']*np.log(np.std(r)))
        for t in range(0,n-1):
            q = params['nu']/2 * (X[t] / np.exp(f[t])-1) - 0.5 + (r[t]**2)/(2*np.exp(f[t]))
            f[t+1]  = params['omega'] +params['beta']*f[t]+params['alpha']*q
        if not annual:
            return f
        else:
            sigma = np.sqrt(252*np.exp(f))
            return sigma[sigma>0]

    def llikhood(parvals):
        params = dict(zip(parkeys,parvals))

        f = model(params)
        L1 = -np.log(gamma(params['nu']/2)) - (params['nu']/2)*np.log(2*np.exp(f)/params['nu'])
        L2 = (params['nu']/2-1) * np.log(X) - (params['nu']*X)/(2*np.exp(f))
        llikhood = np.nanmean(L1+L2)
        if not np.isfinite(llikhood):
            return np.inf
        return -llikhood

    def optim():
        Lprime = lambda x: approx_fprime(x, llikhood, 0.01)  

        est = minimize(llikhood, x0=par_ini,
                       options = {'eps':1e-10,'maxiter':2000},
                       method = 'Newton-CG',jac=Lprime,
                       bounds = [(0,10),(0,10),(0,10),(0,10),(2,10)])
        estimates = est.x
        print(est.fun)
        print(estimates)
        return estimates


    est = optim()
    est = dict(zip(parkeys,est))
    plt.plot(model(est,annual=True),lw=1)
    return model(est,annual=True)



def real_skewedtGAS(daily_returns,RK_values):
    # Student t-RGAS
    r = daily_returns.values.flatten()*100
    X = RK_values.values.flatten()[1:]#*100

    n = len(X)
    par_ini = [ 7.99824816,  3.40016583 , 0.7349543  , 0.00990475 , 0.04243207 ,0.9999085, -0.21990169 , 0.24187858 , 0.86945862]
    parkeys = ['nu1','nu2','gamma','alpha','beta','phi','omega','A','B']



    
    def model(params,annual=False):
        sigma2 = np.zeros(n)
        epsilons = np.zeros(n)
        sigma2[0] = (params['omega']+params['B']*np.log(np.std(r)))
        for t in range(0,n-1):
            
            eps = (np.log(X[t])-params['beta']*np.log(sigma2[t])-params['alpha'])/params['phi']
            nabla1 = 0.5 * (((params['nu1']+1)*r[t]**2)/((params['nu1']-2)*params['gamma']**(2*np.sign(r[t]))*sigma2[t]+r[t]**2)-1)
            nabla2 = params['beta']/params['phi']*((params['nu2']+1)*eps)/((params['nu2']-2)+eps**2)
            sigma2[t+1]=np.exp(params['omega']+params['A']*(nabla1+nabla2)+params['B']*np.log(sigma2[t]))
            
            
            epsilons[t] = eps
        t=len(X)-1
        epsilons[t] = (np.log(X[t])-params['beta']*np.log(sigma2[t])-params['alpha'])/params['phi']
        if not annual:
            return sigma2,epsilons
        else:
            sigma = np.sqrt(252*sigma2)
            return sigma[sigma>0]

    def llikhood(parvals):
        params = dict(zip(parkeys,parvals))

        sigma2,eps = model(params)   
        L1 = np.log(2*gamma((params['nu1']+1)/2) / ((params['gamma']+1/params['gamma'])*np.sqrt((params['nu1']-2)*np.pi)*gamma(params['nu1']/2)))
        L2 = -0.5*np.log(sigma2) - ((params['nu1']+1)/2)*np.log(1+r**2/(params['nu1']-2)*sigma2*params['gamma']**(2*np.sign(r)))
        L3 = np.log(gamma((params['nu2']+1)/2)/(np.sqrt((params['nu2']-2)*np.pi)*gamma(params['nu2']/2)*params['phi']))
        L4 = -((params['nu2']+1)/2) * np.log(1+eps**2/(params['nu2']-2))
        llikhood = np.nanmean(L1+L2+L3+L4)
        
        #print(np.mean(L1),np.mean(L2),np.mean(L3),np.mean(L4))
        if not np.isfinite(llikhood):
            return np.inf
        return -llikhood

    def optim(fit=True):
        Lprime = lambda x: approx_fprime(x, llikhood, 0.01)  
        if fit:
            est = minimize(llikhood, x0=par_ini,
                           options = {'eps':1e-10,'maxiter':2000},
                           method = 'Newton-CG',jac=Lprime)
            estimates = est.x
        else:
            estimates = par_ini
        print(estimates)
        return estimates


    est = optim()
    est = dict(zip(parkeys,est))
    return model(est,annual=True)
