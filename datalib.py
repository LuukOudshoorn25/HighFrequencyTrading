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

class datacleaner():
    def __init__(self):
        tstart=time.time()
        # Get list of all files
        flist = glob('./datafiles/ASML/*csv')
        # Parallel read the files
        dfs= Parallel(n_jobs=8)(delayed(pd.read_csv)(i) for i in flist)
        print("Lengths",[len(w) for w in dfs])
        total_length1 = np.sum([len(w) for w in dfs])
        print(np.round(time.time()-tstart,2),"Read data")
        # Select only exchange "Q"
        dfs= Parallel(n_jobs=8)(delayed(self.select_exchange)(i) for i in dfs)
        print(np.round(time.time()-tstart,2),"Selected on exchange")
        total_length2 = np.sum([len(w) for w in dfs])
        # Drop corrected
        dfs= Parallel(n_jobs=8)(delayed(self.drop_corr)(i) for i in dfs)
        print(np.round(time.time()-tstart,2),"Dropped corrected")
        total_length3 = np.sum([len(w) for w in dfs])
        # Aggregate to second level
        dfs= Parallel(n_jobs=8)(delayed(self.resample)(i) for i in dfs)
        
        # Concatenate them vertically
        self.df = pd.concat(dfs)        
        # Sort them
        self.sort()
        # Get hourly dataframe        
        self.get_hourly()
        # Write to files
        self.write()
        #print(np.round(time.time()-tstart,2), "Done")
        #print(total_length1, total_length2, total_length3, 
        #      total_length4, total_length5, total_length6)
        
    def resample(self,df):
        df = df.reset_index()
        df['DATETIME'] = df['DATE'].astype(str) + ' ' + df['TIME_M']
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df = df.set_index('DATETIME')[['SIZE','PRICE']]
        
        #sizes  = df.resample('1S')[['SIZE']].sum()
        #prices = df.resample('1S')[['PRICE']].median()
        #df = sizes.join(prices)
        return df#.dropna()
        
    def select_exchange(self,df):
        return df[df.EX=='Q']
        
    def neg_outlier(self,df):
        # rolling window to remove outliers at 60std
        prices = df.PRICE
        r = prices.rolling(window=4000)  # Create a rolling object 
        mps = r.mean() - 60. * r.std()  # Combine a mean and stdev on that object
        outliers = prices[prices < mps].index
        df = df.drop(outliers)
        return df
        
    def drop_corr(self,df):
        # Drop all items that are corrected (see Barndorff-Nielsen)
        df = df[df.TR_CORR==0]
        return df
        
        
        
    def sort(self):
        self.df = self.df.sort_index()
        
    def get_hourly(self):
        # Aggregate to hourly dateframe, just for visualization
        self.hourly=self.df.reset_index().resample('H', on='DATETIME').median()
        
    def write(self):
        # Write to pickle files
        self.df.to_pickle('./datafiles/ASML_2015_2020.pickle')
        self.hourly.to_pickle('./datafiles/ASML_hourly.pickle')
    
    def get_df(self):
        return self.df
        

def makedaily(df):
    df['LOGPRICE'] = np.log(df.PRICE)
    df['DELTALOG'] = df['LOGPRICE'].diff()*100
    closingprices = df[['PRICE']].groupby(df.index.date).apply(lambda x: x.iloc[[-1]])
    closingprices.index = closingprices.index.droplevel(0)
    daily_returns = np.log(closingprices).diff().iloc[1:]
    return daily_returns,closingprices