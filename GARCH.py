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

class estimate_GARCH():
    def __init__(self,model,p,q,maxdate=None):# or 'RealGARCH'
        self.options = {'eps':1e-09,
                        'maxiter':2000}
                        
        self.p,self.q = p,q
        self.model = model # 'RealGARCH'
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
        if self.maxdate:
            self.RVOL = self.RVOL['2015':self.maxdate]
        self.RVOL = self.RVOL.dropna().values.flatten()[1:]
        """
        
        self.RVOL = self.RK_values
        if self.maxdate:
            self.RVOL = self.RVOL['2015':self.maxdate]
        self.RVOL = self.RVOL.values.flatten()[1:]"""
    
    def __one_step__(self):
        # Obtain historical sigmas (on which the model was trained)
        # and obtain past returns
        _,conditional_sigma = self.return_vola()
        x = self.closingreturns
        # Get fitted parameters
        params = self.estimates
        omega = np.exp(params[0])
        
        alphas = np.zeros(self.q)
        betas  = np.zeros(self.p)
        
        for i in range(0,self.q):
            alphas[i] = np.exp(params[i+1])/(1+np.exp(params[i+1]))
        for i in range(0,self.p):
            betas[i] = np.exp(params[i+self.q+1])/(1+np.exp(params[i+self.q+1]))
        # Predict one new value
        alpha_part = 0
        beta_part  = 0
        # Obtain beta part (lagged sigma2)
        t = len(conditional_sigma)
        for i in range(0,self.p):
            beta = betas[i]
            beta_part = beta_part+beta*conditional_sigma[t-i-1]
            # Obtain alpha part (lagged returns)
            for i in range(0,self.q):
                alpha = alphas[i]
                if self.model == 'GARCH':
                    alpha_part = alpha_part+alpha*x[t-i-1]**2
                if self.model == 'RealGARCH':
                    alpha_part = alpha_part+alpha*self.RVOL[t-i-1]
            # Combine in sigma2[t]
            sigma_future = omega + alpha_part + beta_part
        return sigma_future
    
    
    
    
    def __llik_fun_GARCH__(self,params,estimate=True):
        x = self.closingreturns
        n = len(x)
        # Convert parameters back from their log normalization
        omega = np.exp(params[0])
        
        alphas = np.zeros(self.q)
        betas  = np.zeros(self.p)
        
        for i in range(0,self.q):
            alphas[i] = np.exp(params[i+1])/(1+np.exp(params[i+1]))
        for i in range(0,self.p):
            betas[i] = np.exp(params[i+self.q+1])/(1+np.exp(params[i+self.q+1]))
        
        # Iterate through sigma2 using the GARCH updating rules
        sigma2 = np.zeros(n)
        # fill first values with sample variance
        for i in range(0,max(self.p,self.q)+1):
            sigma2[i] = np.var(x)
        # Iterate through times
        for t in range(max(self.p,self.q),n):
            alpha_part = 0
            beta_part  = 0
            # Obtain beta part (lagged sigma2)
            for i in range(0,self.p):
                beta = betas[i]
                beta_part = beta_part+beta*sigma2[t-i-1]
            # Obtain alpha part (lagged returns)
            for i in range(0,self.q):
                alpha = alphas[i]
                
                if self.model == 'GARCH':
                    alpha_part = alpha_part+alpha*x[t-i-1]**2
                if self.model == 'RealGARCH':
                    alpha_part = alpha_part+alpha*self.RVOL[t-i-1]
            # Combine in sigma2[t]
            sigma2[t] = omega + alpha_part + beta_part
        
        
        # Derive likelihood
        if estimate:
            L =  -0.5*np.log(2*np.pi) - 0.5*np.log(sigma2) - 0.5*x**2/sigma2
            if self.model == 'RealGARCH':
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
        b = np.ones(self.p)*(0.4/self.p)  # initial value for beta
        a = np.ones(self.q)*(0.1/self.q) # initial value for alpha
        omega = np.nanvar(self.closingreturns)*(1-np.sum(a)-np.sum(b)) # initial value for omega
        
        par_ini = np.array([omega])#np.array([np.log(omega)])
        
        alphas = np.zeros(self.q)
        betas  = np.zeros(self.p)
        
        for i in range(0,self.q):
            alphas[i] = np.log(a[i]/(1-a[i]))
        for i in range(0,self.p):
            betas[i] = np.log(b[i]/(1-b[i]))
        
        par_ini = np.hstack((par_ini,alphas, betas))
        
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
        
        betas = np.array([np.exp(w)/(1+np.exp(w)) for w in self.estimates[1:self.p+1]])
        alphas = np.array([np.exp(w)/(1+np.exp(w)) for w in self.estimates[1+self.p:]])
        
        self.thetahat = np.hstack((omega_hat,alphas,betas))
        
        
    def return_vola(self):
        sigma2 = self.__llik_fun_GARCH__(self.estimates,estimate=False)
        return self.datetimes,sigma2
    
    def return_llik_AIC_BIC(self):
        return self.AIC,self.llik_opt



class parallel_GARCH_fitter():
    def __init__(self):
        self.fittedGARCHES,self.garch_AIC_BIC_llikhood=self.simultaneous_fit(modelfamily='GARCH')
        self.fittedRealGARCHES,self.Realgarch_AIC_BIC_llikhood=self.simultaneous_fit(modelfamily='RealGARCH')
        self.concat()
        self.return_vars()
        
    def concat(self):
        self.allGARCH_AIC_llikhood = pd.concat((self.garch_AIC_BIC_llikhood,self.Realgarch_AIC_BIC_llikhood))
        
    def return_vars(self) :
        return self.fittedGARCHES, self.fittedRealGARCHES, self.allGARCH_AIC_llikhood
    
        
    def exists(self,name):
        return (name in sys._getframe(1).f_locals  # caller's locals
             or name in sys._getframe(1).f_globals # caller's globals
        )

    def simultaneous_fit(self,modelfamily='GARCH'):
        pqpairs = []
        for i in range(1,4):
            for j in range(1,4):
                pqpairs.append((i,j))
        if self.exists('fittedGARCHES') and modelfamily=='GARCH':
            # See if we did this already
            toplot = fittedGARCHES
        elif self.exists('fittedRealGARCHES') and modelfamily=='RealGARCH':
            toplot = fittedRealGARCHES
        else:
            # if not, fit them all simultaneous
            toplot = Parallel(n_jobs=8)(delayed(estimate_GARCH)(model=modelfamily,p=w[0],q=w[1]) for w in pqpairs)
        # Start plotting
        fig,ax=plt.subplots(figsize=(3.321,3.5))
        for i,model in enumerate(toplot):
            x,y = model.return_vola()
            y = np.sqrt(252*y)
            ax.plot(x,y+100*i,lw=0.5,color='black')
            plt.annotate(modelfamily+str(pqpairs[i]),xy=(x[-1]+datetime.timedelta(days=15),y[0]+100*i+3),size=7)
            plt.axhline(100*i,ls='--',lw=0.4,color='tomato')
        deltatime = datetime.datetime(2022,11,16,16,15,0) if modelfamily=='GARCH' else datetime.datetime(2023,6,16,16,15,0)
        ax.set_xlim(datetime.datetime(2015,1,1,9,35,0),deltatime)
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility (% p.a.)')
        ax.set_ylim(0,950)
        ax.set_xticks(ax.get_xticks()[:-1])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('all'+modelfamily+'models.pdf',bbox_inches='tight')
        df=pd.DataFrame({'AIC':[],'llikhood':[]})
        for i,model in enumerate(toplot):
            modelname = modelfamily+str(pqpairs[i])
            df.loc[modelname]=model.return_llik_AIC_BIC()
        return toplot,df




# Now predict one step ahead
class GARCH_RealGARCH_predict():
    def __init__(self,modelfamily,RK_values):
        self.modelfamily = modelfamily
        self.get_pqpairs()
        self.RK_values = RK_values
        # Create an array of future dates. We iterate over these days, fit a model and predict the first 
        # next dates volatility
        self.get_dates()
        self.quickfit = False
        
    def get_pqpairs(self):
        # Get list of all modelsup to (3,3)
        pqpairs = []
        for i in range(1,3):
            for j in range(1,3):
                pqpairs.append((i,j))
        self.pqpairs = pqpairs
        
        
    def quick_fit(self,modelfamily='GARCH',end_day=None):
        # Fit one model for the whole period and use it for predictions
        pqpairs = self.pqpairs
        if not self.quickfit:
            self.estimated_models = Parallel(n_jobs=6)(delayed(estimate_GARCH)(model=modelfamily,p=w[0],q=w[1],maxdate='2020-12-31') for w in pqpairs)
            self.quickfit = True
        
        point_ests = np.zeros(len(self.estimated_models))
        for i,model in enumerate(self.estimated_models):
            model.maxdate = end_day
            model.init_data()
            point_ests[i] = model.__one_step__()
        return point_ests
        
        
        
    def simultaneous_fit(self,modelfamily='GARCH',end_day=None):
        
        N = np.where(np.array(self.end_days) == end_day)[0][0]
        
        eta = ((time.time() - self.starttime)/(N+1))*(len(self.end_days)-N)
        print('Fittig models with dates up to',end_day, 'ETA',np.round(eta,1), ' sec')
        # Get list of all modelsup to (3,3)
        pqpairs = self.pqpairs
        # Fit all models with data up to the end date passed in the function
        estimated_models = [estimate_GARCH(model=modelfamily,p=w[0],q=w[1],maxdate=end_day) for w in pqpairs]
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
            predictions = Parallel(n_jobs=6)(delayed(self.simultaneous_fit)(self.modelfamily,day) for day in self.end_days)
            self.predictions = np.array(predictions)
        else:
            self.predictions = np.array([self.quick_fit(self.modelfamily,day) for day in self.end_days])
        return self.predictions

    def evaluate(self):
        # compare predictions with Realized Kernel volatilities
        # Make dataframe to store results
        results_df = pd.DataFrame({'True',})
        acronyms = ['Pred_'+self.modelfamily+str(w) for w in self.pqpairs]
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

    
def make_GARCH_prediction_table(scores_GARCH,scores_RealGARCH,predictor):
    concat = pd.concat((scores_GARCH,scores_RealGARCH),axis=1)
    concat.index = [str(w) for w in 2*predictor.pqpairs]

    concat = concat.reset_index().groupby('index').mean()
    concat.columns = ['MAE_G','RMSE_G','MAE_RG','RMSE_RG']

    concat = concat.round(2)#print(concat.round(2).to_latex(bold_rows=True))
    print(concat)
    return concat