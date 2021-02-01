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
from scipy.optimize import approx_fprime

class special_GARCH():
    def __init__(self,RK_values,daily_returns,model,p,q,maxdate=None, real=False):
        self.options = {'eps':1e-09,
                      'maxiter':2000}
        self.RK_values = RK_values
        self.daily_returns = daily_returns
        self.p,self.q = p,q
        self.model = model # i.e. E for EGARCH
        self.real = real # use realized measure for VOL (boolean)
        # store the maxdata parameter for if we do predictions
        self.maxdate = maxdate
        self.init_data()
        # Call fit of GARCH
        self.fit_garch(self.closingreturns)
        return
        
    def init_data(self):
        if not self.maxdate:
            df = pd.read_hdf('./datafiles/daily_returns.h5')
            self.datetimes = df.index
            self.closingreturns = df.values.flatten()*100
        else: 
            df = pd.read_hdf('./datafiles/daily_returns.h5')['2015':self.maxdate]
            self.datetimes = df.index
            self.closingreturns = df.values.flatten()*100
        self.__RVOL__()
        
    def __RVOL__(self):
        """Obtain realized volatilities from highfreq data"""
        # Convert to logprices
        self.RVOL = pd.read_hdf('./datafiles/RVOL_parallel_GARCH.hdf')
        # Daily returns do not have first day and no weekends, so drop them
#         if self.maxdate:
#             self.RVOL = self.RVOL['2015':self.maxdate]
#         self.RVOL = self.RVOL.dropna().values.flatten()[1:]

        self.RVOL = self.RK_values
        if self.maxdate:
            self.RVOL = self.RVOL['2015':self.maxdate]
        self.RVOL = self.RVOL.values.flatten()[1:]
    
    def GARCH(self, y, params, conditional_sigma=None):
        if self.real == True:
            y = self.RVOL
        # if predict
        if conditional_sigma is not None:
            # Obtain beta part (lagged sigma2)
            t = len(conditional_sigma)
            alpha_part = 0
            beta_part = 0
            for i in range(0, self.p):
                alpha = 'alpha' + str(i)
                alpha_part = alpha_part + params[alpha]*y[t-1]**2
            for i in range(0, self.q):
                beta = 'beta' + str(i)
                beta_part = beta_part + params[beta]*conditional_sigma[t-1]
            sigma_future = params['mu'] + alpha_part + beta_part
            return sigma_future
    
        else:
            n = len(y)
            ft = np.zeros(n)

            # fill first values with sample variance
            for i in range(0,max(self.p,self.q)+1):
                ft[i] = np.var(y)

            for t in range(max(self.p,self.q),n):
                alpha_part = 0
                beta_part = 0
                for i in range(0, self.p):
                    alpha = 'alpha' + str(i)
                    alpha_part = alpha_part + params[alpha]*y[t-1]**2
                for i in range(0, self.q):
                    beta = 'beta' + str(i)
                    beta_part = beta_part + params[beta]*ft[t-1]
                ft[t] = params['mu'] + alpha_part + beta_part
            return ft
    
    def EGARCH(self, y, params, conditional_sigma=None):
        if self.real == True:
            y = self.RVOL
        # if predict
        if conditional_sigma is not None:
            # Obtain beta part (lagged sigma2)
            t = len(conditional_sigma)
            theta_part = 0
            beta_part = 0
            for i in range(0,self.p):
                name = 'beta' + str(i)
                beta_part = beta_part+params[name]*np.log(conditional_sigma[t-i-1])
            # Obtain alpha part (lagged returns)
            for i in range(0,self.q):
                name = 'theta' + str(i)
                theta_part = theta_part+params[name]*y[t-i-1]
            g = theta_part + np.abs(y[t-1])/np.sqrt(conditional_sigma[t-1])-np.sqrt(2/np.pi)
            sigma_future = np.exp(params['mu'] + params['alpha']*g + beta_part)
            return sigma_future
    
        else: 
            n = len(y)
            ft = np.zeros(n)
            # fill first values with sample variance
            for i in range(0,max(self.p,self.q)+1):
                ft[i] = np.exp(params['mu']+params['alpha']*np.log(np.std(y)))
            # Iterate through times
            for t in range(max(self.p,self.q),n):
                theta_part = 0
                beta_part  = 0
                # Obtain beta part (lagged sigma2)
                for i in range(0,self.p):
                    name = 'beta' + str(i)
                    beta_part = beta_part+params[name]*np.log(ft[t-i-1])
                # Obtain alpha part (lagged returns)
                for i in range(0,self.q):
                    name = 'theta' + str(i)
                    theta_part = theta_part+params[name]*y[t-i-1]

                # Combine in ft[t]
                g = theta_part + np.abs(y[t-1])/np.sqrt(ft[t-1])-np.sqrt(2/np.pi)
                ft[t] = np.exp(params['mu'] + params['alpha']*g + beta_part)
        
            return ft
    
    def TGARCH(self, y, params, conditional_sigma=None):
        if self.real == True:
            y = self.RVOL
        # if predict
        if conditional_sigma is not None:
            # Obtain beta part (lagged sigma2)
            t = len(conditional_sigma) 
            alpha_part = 0
            beta_part = 0
            for i in range(0, self.p):
                alphaP = 'alphaP' + str(i)
                alphaM = 'alphaM' + str(i)
                alpha_part = alpha_part + params[alphaP]*(np.abs(y[t-i-1]) - params[alphaM]*y[t-i-1]) 
            for i in range(0, self.q):
                beta = 'beta' + str(i)
                beta_part = beta_part + params[beta]*np.sqrt(conditional_sigma[t-i-1])
            sigma_future = (params['mu'] + alpha_part + beta_part)**2
            return sigma_future
        
        else:
            n = len(y)
            ft = np.zeros(n)

            # fill first values with sample variance
            for i in range(0,max(self.p,self.q)+1):
                ft[i] = np.var(y)

            for t in range(max(self.p,self.q),n):
                alpha_part = 0
                beta_part = 0
                for i in range(0, self.p):
                    alphaP = 'alphaP' + str(i)
                    alphaM = 'alphaM' + str(i)
                    alpha_part = alpha_part + params[alphaP]*(np.abs(y[t-i-1]) - params[alphaM]*y[t-i-1]) 
                for i in range(0, self.q):
                    beta = 'beta' + str(i)
                    beta_part = beta_part + params[beta]*np.sqrt(ft[t-i-1])
                ft[t] = (params['mu'] + alpha_part + beta_part)**2
            
            return ft   
    
    def QGARCH(self, y, params, conditional_sigma=None):
        if self.real == True:
            y = self.RVOL
        # if predict
        if conditional_sigma is not None:
            # Obtain beta part (lagged sigma2)
            t = len(conditional_sigma) 
            alpha_part = 0
            beta_part = 0
            for i in range(0, self.p):
                alpha = 'alpha' + str(i)
                alpha_part = alpha_part + params[alpha]*y[t-i-1]**2 
            for i in range(0, self.q):
                beta = 'beta' + str(i)
                beta_part = beta_part + params[beta]*conditional_sigma[t-i-1]
            sigma_future = params['mu'] + alpha_part + beta_part + params['phi']*y[t-i-1]
            return sigma_future
    
        else:
            n = len(y)
            ft = np.zeros(n)

            # fill first values with sample variance
            for i in range(0,max(self.p,self.q)+1):
                ft[i] = np.var(y)

            for t in range(max(self.p,self.q),n):
                alpha_part = 0
                beta_part = 0
                for i in range(0, self.p):
                    alpha = 'alpha' + str(i)
                    alpha_part = alpha_part + params[alpha]*y[t-i-1]**2 
                for i in range(0, self.q):
                    beta = 'beta' + str(i)
                    beta_part = beta_part + params[beta]*ft[t-i-1]

                ft[t] = params['mu'] + alpha_part + beta_part + params['phi']*y[t-i-1]

            return ft
    
    def MGARCH(self, y, params, conditional_sigma=None):
        
        if self.real == True:
            y = self.RVOL
        # if predict
        if conditional_sigma is not None:
            # Obtain beta part (lagged sigma2)
            t = len(conditional_sigma)
            at = y[t-1] - params['omega'] - params['c']*conditional_sigma[t-1] 
            alpha_part = 0
            beta_part = 0
            for i in range(0, self.p):
                alpha = 'alpha' + str(i)
                alpha_part = alpha_part + params[alpha]*at**2 
            for i in range(0, self.q):
                beta = 'beta' + str(i)
                beta_part = beta_part + params[beta]*conditional_sigma[t-i-1]
            sigma_future = params['mu'] + alpha_part + beta_part
            return sigma_future
            
        else: 
            n = len(y)
            ft = np.zeros(n)

            # fill first values with sample variance
            for i in range(0,max(self.p,self.q)+1):
                ft[i] = np.var(y)

            for t in range(max(self.p,self.q),n):
                alpha_part = 0
                beta_part = 0
                at = y[t-1] - params['omega'] - params['c']*ft[t-i-1] 
                for i in range(0, self.p):
                    alpha = 'alpha' + str(i)
                    alpha_part = alpha_part + params[alpha]*at**2 
                for i in range(0, self.q):
                    beta = 'beta' + str(i)
                    beta_part = beta_part + params[beta]*ft[t-i-1]

                ft[t] = params['mu'] + alpha_part + beta_part
        
            return ft

    def __one_step__(self):
        # Obtain historical sigmas (on which the model was trained)
        # and obtain past returns
        _,conditional_sigma = self.return_vola()
        x = self.closingreturns
        # Get fitted parameters
        params = dict(zip(self.parameter_keys, self.estimates))  

        # Predict one new value
        if self.model == 'G': # Normal GARCH
            sigma_future = self.GARCH(x, params, conditional_sigma)
        
        if self.model == 'E': # E-GARCH
            sigma_future = self.EGARCH(x, params, conditional_sigma)
            
        elif self.model == 'T': # T-GARCH
            sigma_future = self.TGARCH(x, params, conditional_sigma)
        
        elif self.model == 'Q': # Q-GARCH
            sigma_future = self.QGARCH(x, params, conditional_sigma)
            
        elif self.model == 'M': # M-GARCH
            sigma_future = self.MGARCH(x, params, conditional_sigma)
        
        return sigma_future
        
    
    def __llik_fun_GARCH__(self,params,estimate=True):
        x = self.closingreturns
          
        params = dict(zip(self.parameter_keys, params))  
        
        if self.model == 'G': # Normal GARCH
            sigma2 = self.GARCH(x, params)
        
        if self.model == 'E': # E-GARCH
            sigma2 = self.EGARCH(x, params)
            
        elif self.model == 'T': # T-GARCH
            sigma2 = self.TGARCH(x, params)
        
        elif self.model == 'Q': # Q-GARCH
            sigma2 = self.QGARCH(x, params)
            
        elif self.model == 'M': # M-GARCH
            sigma2 = self.MGARCH(x, params)
         
        # Derive likelihood
        if estimate:
            L = -0.5*np.log(2*np.pi) - 0.5*np.log(sigma2) - 0.5*x**2/sigma2
            if self.real == True:
                ut,var_ut = self.fit_measurementeq(sigma2)
                L += -0.5*np.log(2*np.pi) - 0.5*np.log(var_ut) - 0.5*ut**2/var_ut

            llik = np.mean(L)

            return -1*llik
        else:
            return sigma2
    
    def fit_measurementeq(self,sigma2):
        fit = np.polyfit(sigma2,self.RVOL,deg=1)
        yhat =  sigma2 * fit[0]+fit[1]
        residuals = self.RVOL - yhat
        return residuals, np.var(residuals)
    
    
    def fit_garch(self,x):
        # Initialize values
        
        if self.model == 'G': # Normal GARCH
            self.parameter_keys = (['mu']) 
            for i in range(0, self.p):
                theta = 'alpha' + str(i)
                self.parameter_keys.append(theta) 
            for i in range(0, self.q):
                beta = 'beta' + str(i)
                self.parameter_keys.append(beta)

            par_ini = np.random.rand(len(self.parameter_keys))+0.5
            bounds_model = (len(self.parameter_keys)-self.q)*[(-np.infty,np.infty)] + [(-1,1)]*(self.q)

        if self.model == 'E': # E-GARCH
            self.parameter_keys = (['mu'] + ['alpha']) 
            for i in range(0, self.p):
                theta = 'theta' + str(i)
                self.parameter_keys.append(theta) 
            for i in range(0, self.q):
                beta = 'beta' + str(i)
                self.parameter_keys.append(beta)

            par_ini = [ 0.09864354,  0.21842432, -0.14892917,  0.9328911 ]
            bounds_model = (len(self.parameter_keys)-self.q)*[(-np.infty,np.infty)] + [(-1,1)]*(self.q)
        
        elif self.model == 'T':
            self.parameter_keys = (['mu'])
            for i in range(0, self.p):
                alphaP = 'alphaP' + str(i)
                alphaM = 'alphaM' + str(i)
                self.parameter_keys.append(alphaP)
                self.parameter_keys.append(alphaM)
            for i in range(0, self.q):
                beta = 'beta' + str(i)
                self.parameter_keys.append(beta)
                
            par_ini = [0.23747126, 0.1628816 , 0.3224849 , 0.75267668]
            bounds_model = (len(self.parameter_keys)-self.q)*[(-np.infty,np.infty)] + [(-1,1)]*self.q
        
        elif self.model == 'Q':
            self.parameter_keys = (['mu'] + ['phi'])
            for i in range(0, self.p):
                alpha = 'alpha' + str(i)
                self.parameter_keys.append(alpha)
            for i in range(0, self.q):
                beta = 'beta' + str(i)
                self.parameter_keys.append(beta)
                
            par_ini = [ 0.46012675, -0.14579136,  0.1305561 ,  0.74799034]
            bounds_model = (len(self.parameter_keys)-self.q)*[(-np.infty,np.infty)] + [(-1,1)]*self.q   
            
        elif self.model == 'M':
            self.parameter_keys = (['c'] + ['mu'] + ['omega'])
            for i in range(0, self.p):
                alpha = 'alpha' + str(i)
                self.parameter_keys.append(alpha)
            for i in range(0, self.q):
                beta = 'beta' + str(i)
                self.parameter_keys.append(beta)
                
            par_ini = [0.10781415, 0.39799021, 0.10626266, 0.3335693 , 0.32294072]
            bounds_model = [(-1,1)] + (len(self.parameter_keys)-self.q-1)*[(0,np.infty)] + [(-1,1)]*self.q  
                       
        Lprime = lambda x: approx_fprime(x, self.__llik_fun_GARCH__, 0.01)        
        est = minimize(self.__llik_fun_GARCH__, x0=par_ini,
                       options = self.options,
                       method = 'Newton-CG',
                       jac = Lprime,
                       bounds =  bounds_model
                      )
        llikhood = -est.fun
        self.AIC = 2*(len(par_ini))-2*llikhood
        self.BIC = (len(par_ini)) * np.log(len(self.closingreturns)) - 2*llikhood
        self.llik_opt = llikhood
        
        self.estimates = est.x
        
    def return_vola(self):
        sigma2 = self.__llik_fun_GARCH__(self.estimates,estimate=False)
        return self.daily_returns.index,sigma2
    
    def return_llik_AIC_BIC(self):
        return self.AIC,self.BIC,self.llik_opt




# Now predict one step ahead
class specialGARCH_predict():
    def __init__(self,RK_values, daily_returns,modeltype, real):
        self.modeltype = modeltype
        self.RK_values = RK_values
        self.daily_returns = daily_returns
        self.real = real
        self.get_pqpairs()
        # Create an array of future dates. We iterate over these days, fit a model and predict the first 
        # next dates volatility
        self.get_dates()
        self.quickfit = False
        
    def get_pqpairs(self):
        # Get list of all modelsup to (3,3)
        pqpairs = []
        for i in range(1,2):
            for j in range(1,2):
                pqpairs.append((i,j))
        self.pqpairs = pqpairs
        
        
    def quick_fit(self,modeltype='E',realized=False,end_day=None):
        # Fit one model for the whole period and use it for predictions
        pqpairs = self.pqpairs
        if not self.quickfit:
            self.estimated_models = Parallel(n_jobs=6)(delayed(special_GARCH)(self.RK_values, self.daily_returns, model=modeltype,p=w[0],q=w[1],maxdate='2020-12-31', real=realized) for w in pqpairs)
            self.quickfit = True
        
        point_ests = np.zeros(len(self.estimated_models))
        for i,model in enumerate(self.estimated_models):
            model.maxdate = end_day
            model.init_data()
            point_ests[i] = model.__one_step__()
        return point_ests
        
    def simultaneous_fit(self,modeltype='E',realized=False,end_day=None):
        
        N = np.where(np.array(self.end_days) == end_day)[0][0]
        
        eta = ((time.time() - self.starttime)/(N+1))*(len(self.end_days)-N)
        print('Fitting models with dates up to',end_day, 'ETA',np.round(eta,1), ' sec')
        # Get list of all modelsup to (3,3)
        pqpairs = self.pqpairs
        # Fit all models with data up to the end date passed in the function
        estimated_models = [special_GARCH(self.RK_values, self.daily_returns,model=modeltype,p=w[0],q=w[1],maxdate=end_day, real=realized) for w in pqpairs]
        # Obtain point estimate for future sigma2
        point_ests = np.zeros(len(estimated_models))
        for i,model in enumerate(estimated_models):
            point_ests[i] = model.__one_step__()
        return point_ests
        
    def get_dates(self):
        self.end_days = [w.strftime('%Y-%m-%d') for w in pd.to_datetime(self.RK_values['2019':].index)]
        
    def sim_fit(self,quick=False):
        # Make sure total number of jobs is less than 24
        if not quick:
            predictions = Parallel(n_jobs=6)(delayed(self.simultaneous_fit)(self.modeltype,self.real, day) for day in self.end_days)
            self.predictions = np.array(predictions)
        else:
            self.predictions = np.array([self.quick_fit(self.modeltype,self.real,day) for day in self.end_days])
        return self.predictions

    def evaluate(self):
        # compare predictions with Realized Kernel volatilities
        # Make dataframe to store results
        results_df = pd.DataFrame({'True',})
        acronyms = ['Pred_'+self.modeltype+ '-GARCH' + str(w) for w in self.pqpairs]
        acronyms = dict(zip(acronyms,[np.array([])]*len(acronyms)))
        acronyms['True']=np.array([])
        acronyms['Date']=np.array([])
        df = pd.DataFrame.from_dict(acronyms)
        df = df.set_index('Date')
        predictions = self.predictions
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