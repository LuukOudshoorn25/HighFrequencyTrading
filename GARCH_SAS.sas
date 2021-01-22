/* read data via start menu, saved in dailyreturns */
data dailyreturns_squared;
    set dailyreturns;
    squared = return**2;
run;
proc gplot data=dailyreturns_squared;
      plot squared*VAR1;
   run;
quit;

   proc autoreg data = dailyreturns outset=garch11;
    /* Estimate GARCH(1,1) with normally distributed residuals with AUTOREG*/
      model return = / garch = ( q=2,p=1 ) ;
   run ;
   quit ;
       /* Estimate GARCH(1,1) with t-distributed residuals with MODEL*/
   proc model data = dailyreturns ;
      parms   df 7.5 arch0 .1 arch1 .2 garch1 .75 ;
      /* mean model */
      return = intercept ;
      /* variance model */
      h.return = arch0 + arch1 * xlag(resid.return **2, mse.return)  +
            garch1*xlag(h.return, mse.return);
      /* specify error distribution */
      errormodel return ~ t(h.return,df);
      /* fit the model */
      fit return / method=marquardt;
   run;
   quit;

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


      /* Estimate Quadratic GARCH (QGARCH) Model */
   proc model data = dailyreturns ;
      parms arch0 .1 arch1 .2 garch1 .75 phi .2;
      /* mean model */
      return = intercept ;
      /* variance model */
      h.return = arch0 + arch1*xlag(resid.return**2,mse.return) + garch1*xlag(h.return,mse.return) +
            phi*xlag(-resid.return,mse.return);
      /* fit the model */
      fit return / method = marquardt fiml ;
   run ;
   quit ;


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


      /* Estimate Threshold Garch (TGARCH) Model */
   proc model data = dailyreturns ;
      parms arch0 .1 arch1_plus .1 arch1_minus .1 garch1 .75 ;
      /* mean model */
      return = intercept ;
      /* variance model */
      if zlag(resid.return) < 0 then
         h.return = (arch0 + arch1_plus*zlag(-resid.return) + garch1*zlag(sqrt(h.return)))**2 ;
      else
         h.return = (arch0 + arch1_minus*zlag(-resid.return) + garch1*zlag(sqrt(h.return)))**2 ;
      /* fit the model */
      fit return / method = marquardt fiml ;
   run ;
   quit ;

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







/* Now with std errors*/
      proc autoreg data= dailyreturns outest = uit1 covout;
      model return =  /  garch=( p=1, q=1,  mean = sqrt);
   run;
   quit;
data std_err1 ; 
  Length Model $10. ; Model = 'GARCH1_1'; 
  set uit1; 
  if _name_ = 'ARCH0'   then DO; PARM = _AH_0   ; std_err = sqrt(_AH_0  ) ; END; 
  if _name_ = 'ARCH1'   then DO; PARM = _AH_1   ; std_err = sqrt(_AH_1  ) ; END; 
  if _name_ = 'GARCH1'  then DO; PARM = _GH_1   ; std_err = sqrt(_GH_1  ) ; END; 
  if _name_ = 'DELTA'   then DO; PARM = _Delta_ ; std_err = sqrt(_Delta_) ; END; 

  if std_err ne . ; 
  Keep Model _name_ std_err _STATUS_ pARM;
run ; 

      /* Estimate GARCH-M Model with PROC AUTOREG */
   proc autoreg data= dailyreturns outest = uit2 covout;
      model return =  /  garch=( p=2, q=2,  mean = sqrt) maxITER=500;
   run;
   quit;
data std_err2 ; 
  Length Model $10. ; Model = 'EGARCH2_2'; 
  set uit2; 
  if _name_ = 'ARCH0'   then DO; PARM = _AH_0  ; std_err = sqrt(_AH_0)   ; END;
  if _name_ = 'ARCH1'   then DO; PARM = _AH_1  ; std_err = sqrt(_AH_1)   ; END;
  if _name_ = 'ARCH2'   then DO; PARM = _AH_2  ; std_err = sqrt(_AH_2)   ; END;
  if _name_ = 'GARCH1'  then DO; PARM = _GH_1  ; std_err = sqrt(_GH_1)   ; END;
  if _name_ = 'GARCH2'  then DO; PARM = _GH_2  ; std_err = sqrt(_GH_2)   ; END;
  if _name_ = 'DELTA'   then DO; PARM = _DeltA_; std_err = sqrt(_Delta_) ; END;

  if std_err ne . ; 
  Keep Model _STATUS_ _name_ std_err pARM;
run ; 

/* Estimate GARCH-M Model with PROC AUTOREG */
   proc autoreg data= dailyreturns outest = uit3 covout;
      model return =  /  garch=( p=3, q=3,  mean = sqrt) maxITER=100;
   run;
   quit;
data std_err3 ; 
  Length Model $10. ; Model = 'GARCH3_3'; 
  set uit3; 
  if _name_ = 'ARCH0'   then DO; PARM = _AH_0  ; std_err = sqrt(_AH_0  ) ; END;  
  if _name_ = 'ARCH1'   then DO; PARM = _AH_1  ; std_err = sqrt(_AH_1  ) ; END;  
  if _name_ = 'ARCH2'   then DO; PARM = _AH_2  ; std_err = sqrt(_AH_2  ) ; END;  
  if _name_ = 'ARCH3'   then DO; PARM = _AH_3  ; std_err = sqrt(_AH_3  ) ; END;  
  if _name_ = 'GARCH1'  then DO; PARM = _GH_1  ; std_err = sqrt(_GH_1  ) ; END;  
  if _name_ = 'GARCH2'  then DO; PARM = _GH_2  ; std_err = sqrt(_GH_2  ) ; END;  
  if _name_ = 'GARCH3'  then DO; PARM = _GH_3  ; std_err = sqrt(_GH_3  ) ; END;  
  if _name_ = 'DELTA'   then DO; PARM = _Delta_; std_err = sqrt(_Delta_) ; END;  

  if std_err ne . ; 
  Keep Model _STATUS_ _name_ std_err pARM;
run ; 


   /* Estimate EGARCH Model with PROC AUTOREG */
   proc autoreg data= dailyreturns outest = uit4 covout;
      model return =  / garch=( q=1, p=1 , type = exp) ;
   run;
   quit;
data std_err4 ; 
  Length Model $10. ;  Model = 'EGARCH1_1'; 
  set uit4; 
  if _name_ = 'EARCH0'   then DO; PARM = _AH_0  ; std_err = sqrt(_AH_0  ) ; END;   
  if _name_ = 'EARCH1'   then DO; PARM = _AH_1  ; std_err = sqrt(_AH_1  ) ; END;   
  if _name_ = 'EGARCH1'  then DO; PARM = _GH_1  ; std_err = sqrt(_GH_1  ) ; END;   
  if _name_ = 'THETA'    then DO; PARM = _THETA_; std_err = sqrt(_THETA_) ; END;   

  if std_err ne . ; 
  Keep Model _name_ std_err _STATUS_ PARM;
run ; 
   /* Estimate EGARCH Model with PROC AUTOREG */
   proc autoreg data= dailyreturns outest = uit5 covout;
      model return =  / garch=( q=2, p=2 , type = exp) ;
   run;
   quit;   
data std_err5 ; 
  Length Model $10. ;  Model = 'EGARCH2_2'; 
  set uit5; 
  if _name_ = 'EARCH0'   then DO; PARM = _AH_0  ; std_err = sqrt(_AH_0  ) ; END;   
  if _name_ = 'EARCH1'   then DO; PARM = _AH_1  ; std_err = sqrt(_AH_1  ) ; END;   
  if _name_ = 'EARCH2'   then DO; PARM = _AH_2  ; std_err = sqrt(_AH_2  ) ; END;   
  if _name_ = 'EGARCH1'  then DO; PARM = _GH_1  ; std_err = sqrt(_GH_1  ) ; END;   
  if _name_ = 'EGARCH2'  then DO; PARM = _GH_2  ; std_err = sqrt(_GH_2  ) ; END;   
  if _name_ = 'THETA'    then DO; PARM = _THETA_; std_err = sqrt(_THETA_) ; END;   

  if std_err ne . ; 
  Keep Model _name_ std_err _STATUS_ PARM;
run ; 
   /* Estimate EGARCH Model with PROC AUTOREG */
   proc autoreg data= dailyreturns outest = uit6 covout;
      model return =  / garch=( q=3, p=3 , type = exp) maxITER=100;
   run;
   quit;
data std_err6 ; 
  Length Model $10. ;  Model = 'EGARCH3_3'; 
  set uit6; 
  if _name_ = 'EARCH0'   then DO; PARM = _AH_0  ; std_err = sqrt(_AH_0  ) ; END;   
  if _name_ = 'EARCH1'   then DO; PARM = _AH_1  ; std_err = sqrt(_AH_1  ) ; END;   
  if _name_ = 'EARCH2'   then DO; PARM = _AH_2  ; std_err = sqrt(_AH_2  ) ; END;   
  if _name_ = 'EARCH3'   then DO; PARM = _AH_3  ; std_err = sqrt(_AH_3  ) ; END;   
  if _name_ = 'EGARCH1'  then DO; PARM = _GH_1  ; std_err = sqrt(_GH_1  ) ; END;   
  if _name_ = 'EGARCH2'  then DO; PARM = _GH_2  ; std_err = sqrt(_GH_2  ) ; END;   
  if _name_ = 'EGARCH3'  then DO; PARM = _GH_3  ; std_err = sqrt(_GH_3  ) ; END;   
  if _name_ = 'THETA'    then DO; PARM = _THETA_; std_err = sqrt(_THETA_) ; END;   

  if std_err ne . ; 
  Keep Model _name_ std_err _STATUS_ PARM;
run ; 

DATA ALLES ; 
   SET std_err1 std_err2 std_err3 std_err4 std_err5 std_err6 ;
RUN ; 

/*END witj STD errors*/




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

ods output Autoreg.garchm_1_1.FinalModel.Results.FitSummary
           =fitsum_garchm_1_1;
ods output Autoreg.garchm_1_2.FinalModel.Results.FitSummary
           =fitsum_garchm_1_2;
ods output Autoreg.garchm2_1.FinalModel.Results.FitSummary
           =fitsum_garchm_2_1;



   /* Estimating multiple GARCH-type models */
title "GARCH family";
proc autoreg data=dailyreturns outest=garch_family COVOUT  maxiter=1000 ;
   garch_1_1 :      model return = / noint garch=(p=1,q=1);
   garch_1_2 :      model return = / noint garch=(p=1,q=2);
   garch_2_1 :      model return = / noint garch=(p=2,q=1);
   garch_2_2 :      model return = / noint garch=(p=2,q=2);
   garch_3_2 :      model return = / noint garch=(p=3,q=2);
   garch_2_3 :      model return = / noint garch=(p=2,q=3);
   garch_3_3 :      model return = / noint garch=(p=3,q=3);


   /*egarch_1_1 :     model return = / noint garch=(p=1,q=1,type=egarch);
   egarch_1_2 :     model return = / noint garch=(p=1,q=2,type=egarch);
   egarch_2_1 :     model return = / noint garch=(p=2,q=1,type=egarch);
   
   
   qgarch_1_1 :     model return = / noint garch=(p=1,q=1,type=qgarch);
   qgarch_1_2 :     model return = / noint garch=(p=1,q=2,type=qgarch);
   qgarch_2_1 :     model return = / noint garch=(p=2,q=1,type=qgarch);
   qgarch_2_2 :     model return = / noint garch=(p=2,q=2,type=qgarch);

   tgarch_1_1 :     model return = / noint garch=(p=1,q=1,type=tgarch);
   tgarch_1_2 :     model return = / noint garch=(p=1,q=2,type=tgarch);
   tgarch_2_1 :     model return = / noint garch=(p=2,q=1,type=tgarch);
   tgarch_2_2 :     model return = / noint garch=(p=2,q=2,type=tgarch);
   */
   */garchm_1_1 :     model return = / noint garch=(p=1,q=1,mean=log);
   */garchm_1_2 :     model return = / noint garch=(p=1,q=2,mean=log);
   */garchm_2_1 :     model return = / noint garch=(p=2,q=1,mean=log);
   
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
	   fitsum_garchm_1_1(In=A17) fitsum_garchm_1_2(In=A18) fitsum_garchm_2_1(In=A19
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
   If A17 then Model_nr = 17 ;
   If A18 then Model_nr = 18 ;
   If A19 then Model_nr = 19 ;

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



data x;
set garch_family;
where _type_ = 'PARM';
run; 


data x ;

  set garch_family ;

  length ParmNaam $8. ;

  where _type_ = 'PARM';

  if _name_ = 'ARCH0'   then DO; PARM1 = _AH_0  ; ParmNaam = '_AH_0'  ; END;

  if _name_ = 'ARCH1'   then DO; PARM = _AH_1  ; ParmNaam = '_AH_1'  ; END;

  if _name_ = 'ARCH2'   then DO; PARM = _AH_2  ; ParmNaam = '_AH_2'  ; END;



  if _name_ = 'EARCH0'  then DO; PARM = _AH_0  ; ParmNaam = '_AH_0'  ; END;

  if _name_ = 'EARCH1'  then DO; PARM = _AH_1  ; ParmNaam = '_AH_1'  ; END;

  if _name_ = 'EARCH2'  then DO; PARM = _AH_2  ; ParmNaam = '_AH_2'  ; END;

  if _name_ = 'EGARCH1' then DO; PARM = _GH_1  ; ParmNaam = '_GH_1'  ; END;

  if _name_ = 'EGARCH2' then DO; PARM = _GH_2  ; ParmNaam = '_GH_2'  ; END;

  if _name_ = 'GARCH1'  then DO; PARM = _GH_1  ; ParmNaam = '_GH_1'  ; END;

  if _name_ = 'GARCH2'  then DO; PARM = _GH_2  ; ParmNaam = '_GH_2'  ; END;

  if _name_ = 'THETA'   then DO; PARM = _THETA_; ParmNaam = '_THETA_'; END;

 

  if _name_ = 'QARCHA0' then DO; PARM = _AH_0  ; ParmNaam = '_AH_0'  ; END;

  if _name_ = 'QARCHA1' then DO; PARM = _AH_1  ; ParmNaam = '_AH_1'  ; END;

  if _name_ = 'QARCHA2' then DO; PARM = _AH_2  ; ParmNaam = '_AH_2'  ; END;

  if _name_ = 'QARCHB1' then DO; PARM = _AHQ_1 ; ParmNaam = '_AHQ_1' ; END;

  if _name_ = 'QARCHB2' then DO; PARM = _AHQ_2 ; ParmNaam = '_AHQ_2' ; END;

  if _name_ = 'QGARCH1' then DO; PARM = _GH_1  ; ParmNaam = '_GH_1'  ; END;

  if _name_ = 'QGARCH2' then DO; PARM = _GH_1  ; ParmNaam = '_GH_1'  ; END;

 

  if _name_ = 'TARCHA0' then DO; PARM = _AH_0  ; ParmNaam = '_AH_0'  ; END;

  if _name_ = 'TARCHA1' then DO; PARM = _AH_1  ; ParmNaam = '_AH_1'  ; END;

  if _name_ = 'TARCHA2' then DO; PARM = _AH_2  ; ParmNaam = '_AH_2'  ; END;

  if _name_ = 'TARCHB1' then DO; PARM = _AHQ_1 ; ParmNaam = '_AHQ_1' ; END;

  if _name_ = 'TARCHB2' then DO; PARM = _AHQ_2 ; ParmNaam = '_AHQ_2' ; END;

  if _name_ = 'TGARCH1' then DO; PARM = _GH_1  ; ParmNaam = '_GH_1'  ; END;

  if _name_ = 'TGARCH2' then DO; PARM = _GH_1  ; ParmNaam = '_GH_1'  ; END;

 

  Keep _Model_ _name_ _stderr_ _STATUS_ pARM pARMNaam;

run ;





















/* Now we will try RealGARCH! */
/* Estimate RealGARCH(1,1) with generalized error distribution residuals */




/* Estimate RealGARCH Model with PROC MODEL */
proc model data = dailyreturns ;
   parms earch0 .1 earch1 .2 egarch1 .75;
   /* mean model */
   return = intercept ;
   /* variance model */
   if (_obs_ = 1 ) then
      h.return = exp( earch0 + egarch1 * log(mse.return)  );
   else h.return = exp(earch0 + earch1*zlag(nresid.logRV) + egarch1*log(zlag(h.return))) ;
   /* fit the model */
   fit return / fiml method = marquardt ;
run;
quit;


   /* Estimate EGARCH Model with PROC AUTOREG */
   proc autoreg data= dailyreturns ;
      model return =  / garch=( q=1, p=1 , type = exp) ;
   run;
   quit;




   /* Estimate RealGARCH Model with log specification with PROC MODEL */
   proc model data = dailyreturns ;
      parms  earch0 .1 earch1 .2 egarch1 .75;
      /* mean model */
      return = intercept ;
      /* variance model */
      if (_obs_ = 1 ) then
         h.return = exp( earch0 + egarch1 * log(mse.return)  );
      else h.return = exp(earch0 + earch1*zlag(g) + egarch1*log(zlag(h.return))) ;
      g = (logRV) + 0.000000000000000000000000000000000000000000000000000001*abs(-nresid.return) ;
      /* fit the model */
      fit return / fiml method = marquardt ;
   run;
   quit;


   /* Estimate RealGARCH Model with linear specification with PROC MODEL */
   proc model data = dailyreturns maxiter=1000;
      parms  earch0 5 earch1 5 egarch1 8;
      /* mean model */
      return = intercept ;
      /* variance model */
      if (_obs_ = 1 ) then
         h.return = earch0 + egarch1 * mse.return ;
      else h.return = earch0 + earch1*zlag(g) + egarch1*zlag(h.return) ;
      g = (RV) + 0.000000000000000000000000000000000000000000000000000001*abs(-nresid.return) ;
      /* fit the model */
      fit return / fiml method = marquardt ;
   run;
   quit;













proc summary data=dailyreturns;
   var return ;
   output out=uit stddev=;
run;
proc sql;
   select return into :stdev from uit;
run;
%put &stdev;
data uit;
   set uit;
   put return=;
run;
