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


def plot_one_day(df_highres_covid):
    fig,ax = plt.subplots(figsize=(6.4,2))
    ax.scatter(df_highres_covid.index, df_highres_covid.PRICE,s=0.08,color='black')
    
    xformatter = mdates.DateFormatter('%H:%M')
    plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)
    plt.xlabel('Time')
    #plt.yticks(np.arange(135,149,3))
    plt.ylim(212,250)
    plt.ylabel('Stock price ($)')
    
    plt.xlim(datetime.datetime(2020,3,16,9,35,0),datetime.datetime(2020,3,16,16,15,0))
    axins = ax.inset_axes([0.05, 0.05, 0.85, 0.35])
    corona_peak1_min = df_highres_covid.resample('1T')[['PRICE']].median().fillna(method='ffill')
    corona_peak5_min = df_highres_covid.resample('5T')[['PRICE']].median().fillna(method='ffill')
    corona_peak1_sec = df_highres_covid.resample('1s')[['PRICE']].median().fillna(method='ffill')
    
    axins.plot(corona_peak1_sec.index, corona_peak1_sec,lw=0.8,color='orange',label = '1 second')
    #axins.plot(corona_peak1_sec.index, corona_peak1_sec,lw=0.8,color='orange',label='1 second')
    axins.plot(corona_peak1_min.index, corona_peak1_min,lw=0.8,color='tomato',label='1 minute')
    axins.plot(corona_peak5_min.index, corona_peak5_min,lw=0.8,color='dodgerblue',label='5 minutes')
    
    
    
    # sub region of the original image
    x1 = datetime.datetime(2020,3,16,12,0,0)
    x2 = datetime.datetime(2020,3,16,12,10,0)
    y1, y2 = 243,249#144.7,146.8
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    #axins.tick_params(axis="y",direction='in',pad=-35)

    #axins.set_yticks([0.5,0.55])
    axins.set_yticklabels('')
    ax.indicate_inset_zoom(axins,lw=1,alpha=0.8,edgecolor='black',label='')
    handles, labels = axins.get_legend_handles_labels()
    ax.legend(handles, labels,loc='upper right',frameon=1)
    plt.tight_layout()
    plt.savefig('ASML_intraday.pdf',bbox_inches='tight')
    
    plt.show()
def plot_all(daily_returns):
    plt.figure(figsize=(3.321,2))
    plt.plot(daily_returns.index, daily_returns.PRICE,color='black',lw=1)
    
    plt.xlabel('Date')
    plt.ylabel('Stock price ($)')
    plt.tight_layout()
    plt.savefig('ASML_6years.pdf',bbox_inches='tight')
    plt.show()


def plot_RK_ASML_vs_index(RK_values):
    data = pd.read_csv('./datafiles/oxfordmanrealizedvolatilityindices.csv')
    #get S&P500 index
    SP = data.loc[data['Symbol'] == '.AEX'] #.AEX
    SP = SP.rename(columns = {'Unnamed: 0': 'DATETIME'})
    SP = SP.loc[SP['DATETIME'] >= '2015-01-02 00:00:00+00:00']
    SP = SP.loc[SP['DATETIME'] <= '2020-12-31 00:00:00+00:00']

    SP = SP.set_index('DATETIME')
    SP.index = pd.to_datetime(SP.index)
    SP.index = pd.to_datetime([w.strftime('%Y-%m-%d') for w in SP.index])

    parzen_SP500 =np.sqrt(252*1e4*SP[['rk_parzen']])
    parzen_ASML = np.sqrt(252*RK_values)
    fig,[ax1,ax2] = plt.subplots(nrows=2,figsize=(3.321,3.5))
    ax1.plot(parzen_SP500,color='tomato',label='Parzen volatility AEX', linewidth = 0.8)
    ax1.plot(np.sqrt(252*RK_values), color = 'dodgerblue', label = "Parzen volatility ASML", linewidth = 0.8)
    ax1.legend(frameon = 1, loc = 'upper left')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('RKvol (% p.a.)')

    SP_ASML_joined = pd.merge(parzen_ASML,parzen_SP500,left_index=True,right_index=True)
    ax2.scatter(SP_ASML_joined.rk_parzen,SP_ASML_joined.RealizedKernel,s=0.8,color = 'dodgerblue')
    yfit = np.polyfit(SP_ASML_joined.rk_parzen,SP_ASML_joined.RealizedKernel,1)
    xnew = np.arange(SP_ASML_joined.rk_parzen.min(),SP_ASML_joined.rk_parzen.max())
    label=r'$\sigma_{ASML}=$' + str(np.round(yfit[0],2)) + r'$\times \sigma_{AEX}+$' + str(np.round(yfit[1],2))
    ax2.plot(xnew,yfit[0]*xnew+yfit[1],ls='--',color='tomato',lw=0.6,label=label)
    ax2.set_xlabel('AEX RKvol (% p.a.)')
    ax2.set_ylabel('ASML RKvol (% p.a.)')
    ax2.legend(frameon=1,loc='upper left')
    plt.tight_layout(pad=0.3)
    plt.savefig('index_vs_ASML.pdf',bbox_inches='tight')
    plt.show()


def compare_RV_RK(RK_values, RVOL_30s, RVOL_5min, RVOL_60min):
    #RVOL_1s = RV('1S')
    RK = np.sqrt(252*RK_values.RealizedKernel)
    fig,ax = plt.subplots()
    #plt.plot(RVOL_1s.index,RVOL_1s.values,lw=0.6,label='RVOL 1 sec')
    plt.plot(RVOL_30s.index,RVOL_30s.values,lw=1,label='RVOL 30 sec',color='dodgerblue',alpha=0.84)
    plt.plot(RVOL_5min.index,RVOL_5min.values,lw=1,label='RVOL 5 min',color='tomato',alpha=0.84)
    plt.plot(RVOL_60min.index,RVOL_60min.values,lw=1,label='RVOL 60 min',color='green',alpha=0.84)
    plt.plot(RK_values.index, RK,lw=1,label='Realized Kernel',color='black')
    plt.legend(frameon=1,loc='upper left')
    plt.tight_layout()
    plt.xlim([datetime.date(2018, 6, 1), datetime.date(2020, 12, 1)])
    plt.ylabel('Annualized volatility [%]')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.savefig('RVOL_vs_RK.pdf',bbox_inches='tight')
    plt.show()
    plt.show()



import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import t

def return_dist(closingreturns):
    mu_data = np.mean(closingreturns)
    std_data = np.std(closingreturns)
    skew_data = stats.skew(closingreturns)
    kurt_data = stats.kurtosis(closingreturns)

    # Fit a normal distribution to the data
    mu, std = norm.fit(closingreturns)
    p = norm.pdf(closingreturns, mu, std)

    # Fit a t-distribution to the data
    df = t.fit(closingreturns)[0]
    mean, var, skew, kurt = t.stats(df, moments='mvsk')
    
    fig, ax = plt.subplots(1, 1)
    # Plot histogram
    plt.hist(closingreturns, bins=100, density=True, color='black', alpha=0.3, label='Data')

    # Plot the normal PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=1, label='Gaussian', color='dodgerblue')

    # Plot the t PDF
    x = np.linspace(t.ppf(0.0001, df), t.ppf(0.9999, df), 100)
    plt.plot(x, t.pdf(x, df), alpha=1, label='Student-t', color='tomato', lw='1')

    plt.xlabel('Returns ($\%$)')
    plt.ylabel('Density')
    plt.legend(frameon=1)
    plt.tight_layout()
    plt.savefig('Return_distribution.pdf',bbox_inches='tight')
    plt.show()
    print('mean:', mu_data, 'skew:', skew_data, 'kurtosis:', kurt_data)
    



def plot_GARCHestimates(fittedGARCHES, fittedRealGARCHES):
    GARCH11model = fittedGARCHES[0]
    RGARCH11model = fittedRealGARCHES[0]
    GARCH11x,GARCH11y = GARCH11model.return_vola()
    RGARCH11x,RGARCH11y = RGARCH11model.return_vola()
    
    fig,[ax1,ax2]=plt.subplots(nrows=2)
    ax1.plot(GARCH11x,np.sqrt(252*GARCH11y),label='GARCH(1,1)',lw=0.8,color='dodgerblue')
    ax1.plot(RGARCH11x,np.sqrt(252*RGARCH11y),label='RealGARCH(1,1)',lw=0.8,color='tomato')
    
    ratio = np.sqrt(252*RGARCH11y)/np.sqrt(252*GARCH11y)
    meanratio = np.mean(ratio)
    print('Average ratio of realgarch over garch',meanratio)
    
    ax2.plot(GARCH11x,ratio,label='Ratio',lw=0.8,color='grey')
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
    plt.savefig('Garch_RealGARCH_RVOL.pdf',bbox_inches='tight')
    plt.show()




def xt_vs_ht(m_lin, m_loglin,RK_values):
    ht_lin = m_lin.return_vola()[1]
    ht_log = m_loglin.return_vola()[1]

    fig, [ax1,ax2] = plt.subplots(nrows=2,figsize=(3.321,3.5))
    yfit1 = np.polyfit(ht_lin, RK_values.values[1:],1)
    yfit2 = np.polyfit(np.log(ht_log),np.log(RK_values.values[1:]),1)

    label1 = r'$x_t=$'+str(np.round(yfit1[0][0],2))+r'$h_t$'+str(np.round(yfit1[1][0],2))
    label2 = r'$\log\, x_t=$'+str(np.round(yfit2[0][0],2))+r'$\log \,h_t$'+str(np.round(yfit2[1][0],2))
    ax1.scatter(ht_lin, RK_values.values[1:],s=1)

    ax2.scatter(np.log(ht_log),np.log(RK_values.values[1:]),s=1)



    x1 = np.arange(ht_lin.min(),ht_lin.max())
    x2 = np.arange(np.log(ht_log).min(),np.log(ht_log).max())

    ax1.plot(x1,yfit1[0]*x1+yfit1[1],lw=1,ls='--',color='tomato',label=label1)
    ax1.set_xlim(0,30)
    ax1.set_ylim(0,12)

    ax2.plot(x2,yfit2[0]*x2+yfit2[1],lw=1,ls='--',color='tomato',label=label2)

    ax1.legend(frameon=1, loc='upper left')
    ax2.legend(frameon=1, loc='upper left')
    ax2.set_xlabel(r'$\log (h_t)$')
    ax1.set_ylabel(r'$x_t$')
    ax1.set_xlabel(r'$h_t$')
    ax2.set_ylabel(r'$\log (x_t)$')

    plt.tight_layout()
    
    plt.savefig('xt_ht_RealGARCH.pdf',bbox_inches='tight')
    plt.show()




def plot_GASestimates(model1, model2, model3, RK_values, fitted_tgas, fitted_ggas, fitted_skewtgas):
    
    fig,[ax1,ax2,ax3]=plt.subplots(nrows=3, figsize=(3.321,4.2))
    ax1.plot(*model1.return_vola(annual=True),label='t-GAS',lw=0.8,color='black')
    ax2.plot(*model2.return_vola(annual=True),label='G-GAS',lw=0.8,color='black')
    ax3.plot(*model3.return_vola(annual=True),label='skewed_t-GAS',lw=0.8,color='black')
    ax1.plot(RK_values.index[1:],fitted_tgas,label='t-RGAS',lw=0.5,color='red', ls='--')
    ax2.plot(RK_values.index[1:],fitted_ggas,label='G-RGAS',lw=0.5,color='red', ls='--')
    ax3.plot(RK_values.index[1:],fitted_skewtgas,label='skewed_t-RGAS',lw=0.5,color='red', ls='--')
    #ax3.plot(*model3_real.return_vola(annual=True),label='skewed_t-RGAS',lw=0.5,color='red', ls='--')
    ax1.set_xlim([datetime.date(2017, 6, 1), datetime.date(2020, 12, 1)])
    ax1.set_xticklabels(['']*len(ax1.get_xticks()))
    ax2.set_xlim([datetime.date(2017, 6, 1), datetime.date(2020, 12, 1)])
    ax2.set_xticklabels(['']*len(ax3.get_xticks()))
    ax3.set_xlim([datetime.date(2017, 6, 1), datetime.date(2020, 12, 1)])


    plt.xlabel('Date')
    plt.xticks(rotation=45)
    ax1.set_ylabel('Volatility (% p.a.)')
    ax2.set_ylabel('Volatility (% p.a.)')
    ax3.set_ylabel('Volatility (% p.a.)')
    ax1.legend(frameon=True,loc='upper left')
    ax2.legend(frameon=True,loc='upper left')
    ax3.legend(frameon=True,loc='upper left')
    plt.tight_layout(pad=-0.4)
    ax1.set_ylim(-12,140)
    ax2.set_ylim(-12,140)
    ax3.set_ylim(-12,140)
    ax1.axhline(0,lw=0.2,ls='--',color='black')
    ax2.axhline(0,lw=0.2,ls='--',color='black')
    plt.savefig('GAS.pdf',bbox_inches='tight')
    plt.show()
