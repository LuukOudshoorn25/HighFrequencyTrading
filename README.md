# HighFrequencyTrading
This repo contains all work on the High frequency trading project. In the main.py file we call all subclasses from other py files. These include fitting of GARCH models, specialGARCH models, GAS models,. Neural Networks, etc. Also all plotting is included. However, we have not included the high frequence data. We took a snapshot of the COVID period (highfreq) such that most of the plotting of highfreq data still works. We have included the daily data on realized volatilities and returns. Note that the code was designed to run in parallel to optimize many of the tasks. Especially the Bayesian estimation (MCMC) and the fitting (estimting MAE/RMSE) will not work well on a small machine. Note furthermore that our ANN / RNN was designed to run on GPU only. Running it on a system without GPU will thus not work. A fail-safe is implemented that breaks the function without error messages if no Tensorflow Enabled GPU was found.


Instructions:
All code was already set to dummy mode, with less iterations to speed things up a bit
1) Obviously, git pull the repo, the dev(elopment) and papers folders can be ignored
2) run the main.py file in python3 from the root directory. For example: 
        python3 main.py

3) The first plots will appear very soon. Afterwards, it will become slower as we need to fit many models and run the Bayesian / MCMC analysis
4) Enjoy! :-)
5) In the dev folder, there is also still some interesting SAS code and results that were used to verify our own approaches

