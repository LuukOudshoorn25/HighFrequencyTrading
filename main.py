###################################################################
###                                                             ###
###    File composed by Luuk Oudshoorn on behalf of group 18    ###
###             Compatible with standard python3                ###
###      Note: most of the scripts run in parallel. This        ###
###        might cause some problems in case this is not        ###
###           available on the host machine when running        ###
###                                                             ###
###################################################################

# Set this to True to work only on daily data
handin=True
plot=False
# Import some of the required libraries
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

# Import codefiles (our own subfiles)
from datalib import datacleaner, makedaily
from plottinglib import plot_one_day,plot_all, plot_RK_ASML_vs_index, compare_RV_RK, return_dist, plot_GARCHestimates, xt_vs_ht, plot_GASestimates
from volatilities import realized_kernel, RV, signature_plot
from GARCH import estimate_GARCH, parallel_GARCH_fitter, GARCH_RealGARCH_predict,make_GARCH_prediction_table
from log_lin_activation import plot_GARCHestimates,  lin_loglin_realgarch
from mcmc import emcee_class
from GAS import estimate_GAS, GAS_RealGAS_predict, real_skewedtGAS, real_GGAS, real_tGAS
from baseline import baseline_predict
from specialgarch import special_GARCH, specialGARCH_predict
from neuralnet import RNN

# Load stylesheet for fancy plotting
plt.style.use('MNRAS_stylesheet')

# Load high frequency data
if not handin:
    # Check if the data is already there
    try:
        # Maybe the large pickle file?
        df = pd.read_pickle('./datafiles/ASML_2015_2020.pickle')
    except:
        try:
            # Maybe the two CSVs?
            df = pd.concat([pd.read_csv(w) for w in ['./datafiles/df.part1.csv','./datafiles/df.part2.csv']])
        except:
            # Ok, none. Lets read in the data from the raw files
            reader = datacleaner()
            df = reader.get_df()

# Load highfreq data for COVID day
df_highres_covid = pd.read_pickle('./datafiles/highres_covid.pickle')

# Convert to daily data
# Check if the data is already there
try:
    daily_returns = pd.read_hdf('./datafiles/daily_returns.h5')
    closingprices  = pd.read_pickle('./datafiles/closingprices.pickle')
except:
    daily_returns,closingprices = makedaily(df)

# Plot one day
if plot:
    plot_one_day(df_highres_covid)
    plot_all(closingprices)

# Obtain realized kernel
if handin:
    RK_values = pd.read_pickle('./datafiles/asml_RK.pickle')
if not handin:
    RK = realized_kernel(df)
    RK_values,gamma_1s = RK.iterate_over_days()
    RK_values = RK_values.set_index('Day')
    RK_values.index = pd.to_datetime(RK_values.index)
    RK_values.to_pickle('./datafiles/asml_RK.pickle')


 #Plot ASML vs index
if plot:
    plot_RK_ASML_vs_index(RK_values)

# Get Realized volatilities for different sampling frequencies
 # Needs high freq data again so we cannot do this in the handin version 
if not handin:
    RVOL_30s = RV('30S', df)
    RVOL_5min =  RV('5T', df)
    RVOL_60min =  RV('60T', df)
    if plot:
        compare_RV_RK(RK_values, RVOL_30s, RVOL_5min, RVOL_60min)


# Get signature plot (higher sampling frequency is higher volatility)
if not handin:
    sigplotter = signature_plot(df)
    sigplotter.__plot__(df)


# Plot return distrribution
if plot:
    returns_daily = daily_returns.values.flatten()*100
    return_dist(returns_daily)


# Now estimate GARCH model

fittedGARCHES, fittedRealGARCHES, allGARCH_AIC_llikhood = parallel_GARCH_fitter().return_vars()
print(allGARCH_AIC_llikhood)


# Now get prediction qualities (this takes long!! We fit 10,000 models)
# When quick=true is set, we do a quick fit but this is less reliable
predictor = GARCH_RealGARCH_predict('GARCH',RK_values)
predictor.sim_fit(quick=True)
scores_GARCH = predictor.get_scores()


predictor = GARCH_RealGARCH_predict('RealGARCH',RK_values)
predictions = predictor.sim_fit(quick=True)
scores_RealGARCH = predictor.get_scores()
# Make one summary table 

make_GARCH_prediction_table(scores_GARCH,scores_RealGARCH,predictor)


# Plot GARCH estimates
#plot_GARCHestimates(fittedGARCHES, fittedRealGARCHES)

# Compare log and linear updating equaton (heteroskedasticity)
# Fit real garch with log and with lin updating function
m_lin = lin_loglin_realgarch('linear')
m_loglin = lin_loglin_realgarch('log-linear')
# Plot the results
plot_GARCHestimates(m_lin, m_loglin)
# Plot xt vs ht
xt_vs_ht(m_lin, m_loglin,RK_values)


# Run MCMC Monte Carlo / Bayesian fitting
# If handin is set, a very small MCMC is run (still it should take 20 sec or so on 22 cores. )
EC = emcee_class(daily_returns,fittedGARCHES, fittedRealGARCHES,modeltype='GARCH22',handin=True)
EC.__corner__()
EC.__chains__()


EC = emcee_class(daily_returns,fittedGARCHES, fittedRealGARCHES,modeltype='EGARCH11',handin=True)
EC.__corner__()
EC.__chains__()


# Estimate GAS model
model1 = estimate_GAS(model='t-GAS')
model2 = estimate_GAS(model='G-GAS')
model3 = estimate_GAS(model='skewed_t')

# We simultaneously again fit all GAS models and get MAE / RMSE (takes very long)
if not handin:
    predictor = GAS_RealGAS_predict(RK_values)
    predictions = predictor.sim_fit()
    scores_GAS = predictor.get_scores()
    scores_GAS


# Estimate real GAS
fitted_tgas = real_tGAS(daily_returns, RK_values)
fitted_ggas = real_GGAS(daily_returns, RK_values)
fitted_skewtgas = real_skewedtGAS(daily_returns, RK_values)

# Plot the results
plot_GASestimates(model1, model2, model3, RK_values, fitted_tgas, fitted_ggas, fitted_skewtgas)

# Get baseline prediction
baseline_predict(np.sqrt(252*RK_values))

# Fit Special GARCH models (EGARCH, TGARCH, etc_)
for m in ['E','Q','M','T']:
    # Loop over the four modeltypes
    # Fit garch
    GARCH = special_GARCH(RK_values,daily_returns,model=m, p=1, q=1, real=False)
    # Obtain time index and volatilities. 
    GARCH_time, GARCH_vola = GARCH.return_vola()

# Get MAE/RMSE for all special GARCH models
results = []
for m in ['E','Q','M','T']:
    # Loop over the four modeltypes
    # Fit predictor class (quick)
    predictor = specialGARCH_predict(RK_values,daily_returns,m, real=False)
    predictor.sim_fit(quick=True)
    scores_specialGARCH = predictor.get_scores()
    results.append(scores_specialGARCH)

print(pd.concat(results))


# Train Recurrent Neural Network 
# Note! This only works on GPU! And thus a system with Tensorflow GPU working and enabled
rnn = RNN()      
