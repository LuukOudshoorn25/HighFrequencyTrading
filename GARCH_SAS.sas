/* read data via start menu, saved in dailyreturns */
data dailyreturns_squared;
    set dailyreturns;
    squared = return**2;
run;
proc gplot data=dailyreturns_squared;
      plot squared*num;
   run;
quit;

   proc autoreg data = dailyreturns ;
    /* Estimate GARCH(1,1) with normally distributed residuals with AUTOREG*/
      model return = / garch = ( q=1,p=1 ) ;
   run ;
   quit ;
   proc autoreg data = dailyreturns ;
    /* Estimate GARCH(1,1) with normally distributed residuals with AUTOREG*/
      model return = / garch = ( q=2,p=2 ) ;
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
       /* Estimate GARCH(2,2) with t-distributed residuals with AUTOREG*/
   proc autoreg data = dailyreturns ;
      model return = / garch=( q=2, p=2 ) dist = t ;
   run ;
   quit;

       /* Estimate GARCH(3,3) with t-distributed residuals with AUTOREG*/
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
      /* Estimate GARCH-M Model with PROC AUTOREG */
   proc autoreg data= dailyreturns ;
      model return =  /  garch=( p=2, q=2,  mean = sqrt);
   run;
   quit;
      /* Estimate GARCH-M Model with PROC AUTOREG */
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
   quit;   
   /* Estimate EGARCH Model with PROC AUTOREG */
   proc autoreg data= dailyreturns ;
      model return =  / garch=( q=3, p=3 , type = exp) ;
   run;
   quit;



/* Now all of the family */



   
/* Capturing ODS tables into SAS data sets */
ods output Autoreg.garch_1_1.FinalModel.Results.FitSummary
           =fitsum_garch_1_1;
ods output Autoreg.garch_1_2.FinalModel.Results.FitSummary
           =fitsum_garch_1_2;
ods output Autoreg.garch_2_1.FinalModel.Results.FitSummary
           =fitsum_garch_2_1;
ods output Autoreg.garch_2_2.FinalModel.Results.FitSummary
           =fitsum_garch_2_2;


ods output Autoreg.egarch_1_1.FinalModel.Results.FitSummary
           =fitsum_egarch_1_1;
ods output Autoreg.egarch_1_2.FinalModel.Results.FitSummary
           =fitsum_egarch_1_2;
ods output Autoreg.egarch_2_1.FinalModel.Results.FitSummary
           =fitsum_egarch_2_1;
ods output Autoreg.egarch_2_2.FinalModel.Results.FitSummary
           =fitsum_egarch_2_2;

ods output Autoreg.qgarch_1_1.FinalModel.Results.FitSummary
           =fitsum_qgarch_1_1;
ods output Autoreg.qgarch_1_2.FinalModel.Results.FitSummary
           =fitsum_qgarch_1_2;
ods output Autoreg.qgarch_2_1.FinalModel.Results.FitSummary
           =fitsum_qgarch_2_1;
ods output Autoreg.qgarch_2_2.FinalModel.Results.FitSummary
           =fitsum_qgarch_2_2;

ods output Autoreg.tgarch_1_1.FinalModel.Results.FitSummary
           =fitsum_tgarch_1_1;
ods output Autoreg.tgarch_1_2.FinalModel.Results.FitSummary
           =fitsum_tgarch_1_2;
ods output Autoreg.tgarch_2_1.FinalModel.Results.FitSummary
           =fitsum_tgarch_2_1;
ods output Autoreg.tgarch_2_2.FinalModel.Results.FitSummary
           =fitsum_tgarch_2_2;


   /* Estimating multiple GARCH-type models */
title "GARCH family";
proc autoreg data=dailyreturns outest=garch_family;
   garch_1_1 :      model return = / noint garch=(p=1,q=1);
   garch_1_2 :      model return = / noint garch=(p=1,q=1);
   garch_2_2 :      model return = / noint garch=(p=1,q=2);
   garch_2_1 :      model return = / noint garch=(p=2,q=1);

   egarch_1_1 :     model return = / noint garch=(p=1,q=1,type=egarch);
   egarch_1_2 :     model return = / noint garch=(p=1,q=2,type=egarch);
   egarch_2_1 :     model return = / noint garch=(p=2,q=1,type=egarch);
   egarch_2_2 :     model return = / noint garch=(p=2,q=2,type=egarch);
   
   qgarch_1_1 :     model return = / noint garch=(p=1,q=1,type=qgarch);
   qgarch_1_2 :     model return = / noint garch=(p=1,q=2,type=qgarch);
   qgarch_2_1 :     model return = / noint garch=(p=2,q=1,type=qgarch);
   qgarch_2_2 :     model return = / noint garch=(p=2,q=2,type=qgarch);

   tgarch_1_1 :     model return = / noint garch=(p=1,q=1,type=tgarch);
   tgarch_1_2 :     model return = / noint garch=(p=1,q=2,type=tgarch);
   tgarch_2_1 :     model return = / noint garch=(p=2,q=1,type=tgarch);
   tgarch_2_2 :     model return = / noint garch=(p=2,q=2,type=tgarch);

   */garchm_1_1 :     model return = / noint garch=(p=1,q=1,mean=log);
   
   */tgarch_1_1 :     model return = / noint garch=(p=1,q=1,type=tgarch);
   */pgarch_1_1 :     model return = / noint garch=(p=1,q=1,type=pgarch);
run;

/* Printing summary table of parameter estimates */
title "Parameter Estimates for Different Models";
proc print data=garch_family;
   */var _MODEL_ _A_1 _AH_0 _AH_1 _AH_2
       _GH_1  _AHQ_1 _AHT_1 _AHP_1 _THETA_ _LAMBDA_ _DELTA_;
    var _MODEL_ _AH_0 _AH_1 _AH_2
       _GH_1 _GH_2  _THETA_ _AHQ_1 _AHQ_2 _AHT_1 _AHT_2;
run;




data llikhood_aic;
   set fitsum_garch_1_1 (In=A1 ) fitsum_garch_1_2 (In=A2 ) fitsum_garch_2_1 (In=A3 ) fitsum_garch_2_2 (In=A4 )
       fitsum_egarch_1_1(In=A5 ) fitsum_egarch_1_2(In=A6 ) fitsum_egarch_2_1(In=A7 ) fitsum_egarch_2_2(In=A8 )
       fitsum_tgarch_1_1(In=A9 ) fitsum_tgarch_1_2(In=A10) fitsum_tgarch_2_1(In=A11) fitsum_tgarch_2_2(In=A12)
       fitsum_qgarch_1_1(In=A13) fitsum_qgarch_1_2(In=A14) fitsum_qgarch_2_1(In=A15) fitsum_qgarch_2_2(In=A16)
    ;
   If A1  then Model_nr = 1  ;
   If A2  then Model_nr = 2  ;
   If A3  then Model_nr = 3  ;
   If A4  then Model_nr = 4  ;
   If A5  then Model_nr = 5  ;
   If A6  then Model_nr = 6  ;
   If A7  then Model_nr = 7  ;
   If A8  then Model_nr = 8  ;
   If A9  then Model_nr = 9  ;
   If A10 then Model_nr = 10 ;
   If A11 then Model_nr = 11 ;
   If A12 then Model_nr = 12 ;
   If A13 then Model_nr = 13 ;
   If A14 then Model_nr = 14 ;
   If A15 then Model_nr = 15 ;
   If A16 then Model_nr = 16 ;
   where Label1="Log Likelihood" OR Label2="AIC";
   if Label1="Log Likelihood" then do; label = label1 ; type= 'Lik'; Waarde= cValue1; end;
   if Label2="AIC"            then do; label = label1 ; type= 'AIC'; Waarde= cValue2; end;
   keep label type Model  Waarde Model_nr;
run ;

proc sort ; by Model_nr ; run ;

proc transpose data = llikhood_aic out = tr(drop = _name_) ;
   by  Model_nr Model;
   id type ;
   var Waarde ;
run ; 


title "Selection Criteria for Different Models";
proc print data=tr;
   format _NUMERIC_ BEST12.4;
run;
title;
