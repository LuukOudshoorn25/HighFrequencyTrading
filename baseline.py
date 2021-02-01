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



def baseline_predict(x):
    """predict sigma_t for sigma_t+1 each time and evaluate the RMSE and MAE of it
       x is measures of realized volatility in the form of a dateframe"""
    # Get data for 2019 and 2020
    data_2019_2020 = pd.concat((x.loc['2019'],x.loc['2020']))
    # Predict latest value for each next sigma2
    data_2019_2020['predict_last'] = np.roll(data_2019_2020,1)
    # Or predict mean value for each sigma2
    data_2019_2020['predict_mean'] = np.mean(x.RealizedKernel)
    # Remove first since it is not relevant because of rolling
    data_2019_2020 = data_2019_2020.iloc[1:]
    
    todrop = data_2019_2020['2020-02-15':'2020-04-15'].index
    data_2019_2020 = data_2019_2020.drop(todrop)
    
    
    # Obtain RMSE and MAE estimates
    RMSE_last = np.sqrt(mean_squared_error(data_2019_2020.RealizedKernel,data_2019_2020.predict_last))
    MAE_last  = mean_absolute_error(data_2019_2020.RealizedKernel,data_2019_2020.predict_last)
    RMSE_mean = np.sqrt(mean_squared_error(data_2019_2020.RealizedKernel,data_2019_2020.predict_mean))
    MAE_mean  = mean_absolute_error(data_2019_2020.RealizedKernel,data_2019_2020.predict_mean)
    print('BASELINE RMSE, MAE, RMSE, MAE' ,RMSE_last,MAE_last,RMSE_mean,MAE_mean)
    return RMSE_last,MAE_last,RMSE_mean,MAE_mean