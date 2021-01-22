/* read data via start menu, saved in dailyreturns */
data dailyreturns_squared;
    set dailyreturns;
    squared = return**2;
run;
proc gplot data=dailyreturns_squared;
      plot squared*num;
   run;
quit;

    /* Estimate GARCH(1,1) with normally distributed residuals with MODEL*/
   proc model data = dailyreturns ;
       parms arch0 .1 arch1 .2 garch1 .75 ;
       /* mean model */
       return = intercept ;
       /* variance model */
       h.return = arch0 + arch1*xlag(resid.return**2,mse.return) +
             garch1*xlag(h.return,mse.return) ;
       /* fit the model */
       fit return / method = marquardt fiml ;
   run ;
   quit ;

   proc autoreg data = dailyreturns ;
    /* Estimate GARCH(1,1) with normally distributed residuals with AUTOREG*/
      model return = / garch = ( q=3,p=3 ) ;
   run ;
   quit ;

    /* Estimate GARCH(1,1) with t-distributed residuals with AUTOREG*/
   proc autoreg data = dailyreturns ;
      model return = / garch=( q=1, p=1 ) dist = t ;
   run ;
   quit;

       /* Estimate GARCH(1,1) with t-distributed residuals with AUTOREG*/
   proc autoreg data = dailyreturns ;
      model return = / garch=( q=3, p=3 ) dist = t ;
   run ;
   quit;

   /* Estimate GARCH(1,1) with generalized error distribution residuals */
   proc model data = dailyreturns ;
      parms nu 2 arch0 .1 arch1 .2 garch1 .75;
      control mse.return = &var2 ; /*defined in data generating step*/
      /* mean model */
      return = intercept ;
      /* variance model */
      h.return = arch0 + arch1 * xlag(resid.return ** 2, mse.return)  +
            garch1 * xlag(h.return, mse.return);
      /* specify error distribution */
      lambda = sqrt(2**(-2/nu)*gamma(1/nu)/gamma(3/nu)) ;
      obj = log(nu/lambda) -(1 + 1/nu)*log(2) - lgamma(1/nu)-
            .5*abs(resid.return/lambda/sqrt(h.return))**nu - .5*log(h.return) ;
      obj = -obj ;
      errormodel return ~ general(obj,nu);
      /* fit the model */
      fit return / method=marquardt;
   run;
   quit;

   /* Estimate GARCH-M Model with PROC AUTOREG */
   proc autoreg data= dailyreturns ;
      model return =  /  garch=( p=1, q=1,  mean = sqrt);
   run;
   quit;

   /* Estimate GARCH-M (3,3) Model with PROC AUTOREG */
   proc autoreg data= dailyreturns ;
      model return =  /  garch=( p=3, q=3,  mean = sqrt);
   run;
   quit;

   /* Estimate EGARCH Model with PROC AUTOREG */
   proc autoreg data= dailyreturns ;
      model return =  / garch=( q=1, p=1 , type = exp) ;
   run;
   quit;
   /* Estimate EGARCH Model with PROC AUTOREG */
   proc autoreg data= dailyreturns ;
      model return =  / garch=( q=2, p=2 , type = exp) ;
   run;
   quit;   /* Estimate EGARCH Model with PROC AUTOREG */
   proc autoreg data= dailyreturns ;
      model return =  / garch=( q=3, p=3 , type = exp) ;
   run;
   quit;
