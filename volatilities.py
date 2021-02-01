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

class realized_kernel():
    def __init__(self):
        """Initialize, call some functions"""
        self.make_time_index()
        return

    def make_time_index(self):
        """We want the data to be restructured onto a regular 1 sec grid 
           to obtain the sparse estimates. Therefore we built a list of 1sec
           times"""
        # Define time arr for every second
        times = []
        for hours in range(9,16):
            for minutes in range(0,60):
                for seconds in np.arange(0,60,1):
                    h = str(hours) if hours>=10 else '0'+str(hours)
                    m = str(minutes) if minutes>=10 else '0'+str(minutes)
                    s = str(seconds) if seconds>=10 else '0'+str(seconds)
                    times.append(h+':'+m+':'+s)
        #times = times[60:]
        self.times = times[1800:]
        
    def make_regular_df(self,onedaydf):
        """Make 1 sec regular dataframe from irregular one day dataframe of prices.
           Note, irregular_day_df should be a 1-day only dataframe with prices
           (no sizes here). If there is no trade, we take the previous time"""
        # Extract times from datetimes
        onedaydf['logprice'] = np.log(onedaydf.PRICE)
        # Get deltalog and deltalog^2 prices
        onedaydf['deltalog'] = onedaydf.logprice.diff()*100
        onedaydf['deltalog2'] = (onedaydf['deltalog'])**2
        
        regular_df = onedaydf.resample('1S')[['PRICE']].median().fillna(method='ffill')
        regular_df['logprice'] = np.log(regular_df.PRICE)
        # Get deltalog and deltalog^2 prices
        regular_df['deltalog'] = regular_df.logprice.diff()*100
        regular_df['deltalog2'] = (regular_df['deltalog'])**2
        
        return regular_df, onedaydf
        
    
    def get_sparse_IV(self,df):
        """Make 1 minute interpolated dataframe"""
        IV_sparse= []
        for i in range(0,22200):
            ds = datetime.timedelta(seconds=i)
            sec_df = df.loc[df.index[0]+ds:df.index[0]+datetime.timedelta(minutes=20)+ds]
            IV_sparse.append(sec_df['deltalog2'].sum()*6.5*3)
        IV_sparse =np.median(IV_sparse)
        return IV_sparse
        
    
    def get_RV_sparse(self,input_logprices):
        """Obtain estimate for RVsparse using the 1-sec interpolated logprices"""
        offsets = np.arange(1200)
        RVs =  np.zeros(1200) # 20min sets
        for offset in offsets:
            # built array (ie 0,1200,2400, etc)
            tochoose = np.arange(0+offset,len(input_logprices),1200)
            # select each 1200th element (plus offset)
            logprices = input_logprices.iloc[tochoose].values.flatten()
            diffs = np.diff(logprices)*100
            RV = np.sum(diffs**2)
            RVs[offset] = RV
        return np.mean(RVs)
    
    
    def RV_dense(self,input_logprices):
        """Get estimate for the dense RV using all irregular spaced returns"""
        q = 25
        offsets = np.arange(q)
        RVs = np.zeros(q)
        Ns  = np.zeros(q)
        for offset in offsets:
            tochoose = np.arange(0+offset,len(input_logprices),q)
            logprices = input_logprices.iloc[tochoose].values.flatten()
            diffs = np.diff(logprices)*100
            RV = np.sum(diffs**2)
            RVs[offset] = RV
            Ns[offset]  = (diffs>0).sum()
        return RVs,Ns
    
    def get_omega(self,df_irregular):
        """Get estimate for omega^2 hat"""
        # collect RVdense estimates
        RVdenses,Ns = self.RV_dense(df_irregular[['logprice']])
        omega2hat_i = RVdenses / (2*Ns)
        omega2hat = np.mean(omega2hat_i)
        return omega2hat
    
    
    def get_optimal_bandwidth(self,regular_df,irregular_df):
        # get estimate for sparse RV
        rvsparse = self.get_RV_sparse(regular_df[['logprice']])
        # Get omega
        omega2hat = self.get_omega(irregular_df)
        xi2 = omega2hat / rvsparse
        xi = np.sqrt(xi2)
        H =  int(3.5134*len(irregular_df)**(3/5)*xi**(4/5))
        return H
    
    
    def Parzen(self,x):
        """Return Parzen kernel for x"""
        if 0<=np.abs(x)<=1/2:
            return 1-6*np.abs(x)**2+6*np.abs(x)**3
        elif 1/2<=np.abs(x)<=1:
            return 2*(1-np.abs(x))**3
        elif np.abs(x)>1:
            return 0
        
    def obtain_K(self,H,irregular_df):
        # make array to loop from -H to H
        h_arr = np.arange(-H,H+1,1,dtype=int)
        irregular_df['deltalog'] = np.log(irregular_df.PRICE).diff().values*100
        # make flat numpy array from the raw return data (approx second)
        x = irregular_df.deltalog.dropna().values.flatten()
        #x = x[np.abs(x)>0]

        K_X = 0
        parzen_values=[]
        # loop 1 over the lags
        gamma1 = 0
        for h in h_arr:
            j_arr = np.arange(np.abs(h)+1,len(x)+1,dtype=int)
            gamma_h = 0
            # loop two for the autocovariances
            for j in j_arr:
                gamma_h += x[j-1]*x[j-np.abs(h)-1]
            K_X += gamma_h*self.Parzen(h/(H+1))
            if h==1:
                gamma1 = gamma_h
        return K_X,gamma1
    
    def worker(self,day,H=None,get_H=False,ret_gamma1=False):
        """worker to loop over days"""
        try:
            daily_counts = pd.read_hdf('../days.h5')
            counts = daily_counts.loc[day].iloc[0]
            # to be faster: find where we are approximately
            iloc0 = max(int(daily_counts.loc[:day].sum().iloc[0]-counts)-10000,0)
            hdf_df = pd.read_hdf('../data.h5')
            
            oneday = hdf_df.iloc[iloc0:counts+iloc0+10000].loc[day]
            # get regular and irregular dfs
            regular_df, irregular_df = self.make_regular_df(oneday)
            # make sparse (1min) df
            #prse_oneday = self.make_sparse_df(oneday)
            #print(prse_oneday)
            # get RVsparse
            RVsparse = self.get_RV_sparse(regular_df[['logprice']])
            
            # get optimal bandwidth
            Hopt = self.get_optimal_bandwidth(regular_df, irregular_df)
            # get realized kernel estimate 
            if not H:
                H = Hopt
                print('Hopt',Hopt)
            else:
                H = H
            RK_est,gamma1 = self.obtain_K(H,irregular_df)
            print(day,'success')
            if ret_gamma1:
                return RK_est,gamma1
            if get_H:
                return RK_est,H
            else:
                return RK_est
        except:
            print(day,'failed')
            return np.nan
    def iterate_over_days(self):
        # Get unique days in dataset
        daily_counts = pd.read_hdf('../days.h5')
        df = pd.read_hdf('../data.h5')
        days = daily_counts.index[(daily_counts>0).values.flatten()]
        days = [w.strftime('%Y-%m-%d') for w in days]
        # iterate over days
        RK_estimates = Parallel(n_jobs=23)(delayed(self.worker)(i,get_H=False,ret_gamma1=True) for i in days)
        RKs = np.array([w[0] for w in RK_estimates])
        gamma_1s = np.array([w[1] for w in RK_estimates])
        print(gamma_1s)
        #np.array([self.worker(day) for day in days[:N]])#
        self.RKvalues = pd.DataFrame({'Day':days,'RealizedKernel':RKs})
        return self.RKvalues,gamma_1s

    def signature_plot(self,day):
        Hs = np.hstack((np.logspace(0.1,2.5,80,dtype=int),[None]))
        
        #K_Hs = [self.worker(day,H,False,True) for H in Hs]
        K_Hs = Parallel(n_jobs=23)(delayed(self.worker)(day,H,True,False) for H in Hs)
        Ks = np.array([w[0] for w in K_Hs])
        gamma1s = np.array([w[1] for w in K_Hs])

        plt.figure()
        plt.ylabel('IV estimate')

        plt.plot(Hs,Ks,color='dodgerblue',label='Realized Kernel',lw=1,zorder=1)
        #plt.scatter(H_opt,obtain_K(H_opt),color='red',s=2)
        
        
        daily_counts = pd.read_hdf('../days.h5')
        counts = daily_counts.loc[day].iloc[0]
        # to be faster: find where we are approximately
        iloc0 = max(int(daily_counts.loc[:day].sum().iloc[0]-counts)-1000,0)
        hdf_df = pd.read_hdf('../data.h5')
        oneday = hdf_df.iloc[iloc0:counts+iloc0+1000].loc[day]
        
        
        #regular_df, irregular_df = self.make_regular_df(oneday)
        #sparse_IV = self.get_sparse_IV(regular_df)
        sparse_IV = ((np.log(df.loc[day].resample('1T')[['PRICE']].median()).diff()*100)**2).sum().iloc[0]
        plt.axhline(sparse_IV,color='tomato',lw=0.7,label='Sparse volatility')
        plt.xlabel('Bandwidth')
        #plt.ylim(13,26)
        
        plt.scatter(K_Hs[-1][1],K_Hs[-1][0],label=r'$H^*$',color='orange',s=8,zorder=2)
        plt.legend(frameon=1,loc='upper right')
        plt.tight_layout()
        plt.savefig('vola_vs_bandwidth.pdf')
        plt.show()

    def plot_RKvol(self):
        RK_values = self.RKvalues
        ### plot figure
        RK_values.index = pd.to_datetime(RK_values.Day)
        df_1min = df[['PRICE']].resample('1T').median()
        df_1min['logprice'] = np.log(df_1min.PRICE)
        df_1min['deltalog'] = df_1min.logprice.diff()*100
        df_1min['deltalog2'] = (df_1min.logprice.diff()*100)**2
        sparse_volas = np.sqrt(252*df_1min.deltalog2.resample('1D').sum())
        plt.plot(RK_values.index,RK_values.RealizedKernel,lw=1,label='RKVOL')
        plt.plot(sparse_volas[sparse_volas>0],lw=1,label='RVOL')
        plt.legend(frameon=1)
        plt.ylabel('Annualized volatility [%]')


def RV(df, samplefreq):
    # Resample to given frequency
    resampled = df.resample(samplefreq).PRICE.median().dropna()
    resampled['LOGPRICE'] = np.log(resampled)
    resampled['DELTALOG'] = resampled['LOGPRICE'].diff()*100
    resampled['DELTALOG2'] = resampled['DELTALOG']**2
    
    # Obtain RV
    RV = resampled['DELTALOG2'].resample('1D').sum()
    # Obtain RVOL
    return RV
    RVOL = np.sqrt(RV*252)
    RVOL = RVOL[RVOL>0]
    return RVOL




class signature_plot():
    def __init__(self, df):
        # Get df with deltalog returns
        self.length = 100000
        self.__prep_df__(df[:self.length])
        self.__obtain_RVs__()
        return
    
    def __prep_df__(self,df):
        df['LOGPRICE'] = np.log(df.PRICE)
        # Convert to returns
        df['DELTALOG'] = (df.LOGPRICE.diff()*100)
        # Only intraday, no interday returns since these screw up the results
        newday = pd.Series(df.index).diff()[pd.Series(df.index).diff()>pd.Timedelta('1h')].index
        df.loc[df.iloc[newday].index,'DELTALOG']=np.nan
        self.df = df[['DELTALOG']]
        
    def __worker__(self,dT):
        RVs = (self.df[:self.length].DELTALOG**2).resample(str(dT)+'ms').sum()
        RVOL = np.sqrt(np.median(RVs[RVs>0]))
        RVOL = RVOL * np.sqrt(252) * np.sqrt(6.5*3600*1000/dT)
        return RVOL
    
    def __obtain_RVs__(self):
        N = 60
        dT_array = np.logspace(2,6,N,dtype=int)
        RVOLs = Parallel(n_jobs=2)(delayed(self.__worker__)(dT) for dT in dT_array)
        self.dT = dT_array
        self.RVOLs = RVOLs
        
    def __plot__(self):
        plt.figure(figsize=(3.321,2.4))
        plt.scatter(self.dT/60000, self.RVOLs,color='black',s=2)
        plt.semilogx()
        plt.xlabel('Sampling frequency [minutes]')
        plt.ylabel('Annualized volatility [%]')
        plt.tight_layout()
        plt.savefig('signature_plot.pdf',bbox_inches='tight')
        plt.show()
