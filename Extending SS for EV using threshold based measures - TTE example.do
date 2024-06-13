clear all
**** STEP 1: Set-up phase to check LP distribution and to identify survival & censoring distributions *****
* check skew normal distribution
* generate large data with LP values from skew normal distribution
sknor 100000 1234 4.60 0.65 -0.5 5
gen LP = skewnormal
* check visually the distribution and calculate summary statistics
hist LP
summ LP, detail

* identify the baseline rate parameter for the exp survival distribution
* - conditional on the effect of LP being 1 (cal slope = 1)
* trial and error finds that 0.00050 is the one to use
 survsim stime ,  lambdas(0.00050) covariates(LP 1) distribution(exponential) 
 gen dead = 1
* censor at 3 years, the assumed max follow-up
 replace dead = 0 if stime > 3
 replace stime = 3 if stime > 3
 * tell Stata that the data are survival data
stset stime, failure(dead) 
* visualise the survival distribution and check S(3) is as anticipated
sts graph
sts generate St = s
* we see this gives about S(3) = 0.83 and F(3) = 0.17 as desired

* identify the rate for the censoring distribution 
* trial and error finds that a lambda of 0.426 is needed
* as it gives the assumed censoring probabilty of 0.72 by 3 years
clear all
set obs 100000
survsim censtime ,  lambdas(0.426) distribution(exponential) 
 gen cens = 1
* censor at 3 years, the assumed max follow-up, for ease of display
replace cens = 0 if censtime > 3
replace censtime = 3 if censtime > 3
* set the data as survival data
stset censtime, failure(cens) 
* visualise the censoring distribution and S(3)
sts graph
sts generate St = s
* at 3 years we see this gives a censoring probability of about 0.72 as desired


**** STEPS 2 to 9:  simulate data according to a particular size
* then estimate calibration slope and its standard error 
* then repeat over many (e.g. 1000) simulation *****
*** Run all this code in one go from START till END ***
** START **
clear all
tempname testsize
tempfile testfile
postfile `testsize' id slope se_slope events recall se_recall precision se_precision f1 se_f1 spec se_spec accuracy se_accuracy npv se_npv using `testfile', replace
* define our loop, and let us use 1000 simulations  
  qui forvalues i = 1(1)1000 {
	* define LP distribution & sample values for chosen sample size (here 3600)
	sknor 3600 1234 4.60 0.65 -0.5 5
	gen LP = skewnormal
	* generate survival times from the exp dist identified in Step 1
	survsim stime ,  lambdas(0.00050) covariates(LP 1) distribution(exponential) 
	gen dead = 1
	* censor at max follow-up time, say 3 years
	replace dead = 0 if stime > 3
	replace stime = 3 if stime > 3
	* generate censoring times from the exp dist identified in Step 1
	survsim censtime ,  lambdas(0.426) distribution(exponential) 
	replace dead = 0 if censtime < stime
	replace stime = censtime if censtime < stime
	* predicted risk by 3 years using VTE model equation *
	gen Predicted_P = 1 - (0.9983^(exp(LP)))
	* transform to the cumulative log-log scale
	gen cloglog_predicted  = cloglog(Predicted_P)
	* set the data as survival data
	stset stime, failure(dead) 
	* generate pseudo values
	stpci dead , at(3) generate(F_pseudo`i') 
	* fit the calibration model to the validation sample 
	* Estimation method irls is more stable for model convergence than Newton Raphson
	* we use a cloglog link below, but could alternatively use logit
	 glm F_pseudo`i' cloglog_predicted , link(cloglog)  vce(robust) irls
	* extract estimate and SE of the calibration slope
	local slope = e(b)[1,1]
	local se_slope = sqrt(e(V)[1,1])
	* calculate and extract number of events in the validation sample
	count if dead > 0 
	local events = r(N)
	* other measures could also be estimated at this point, such as C, D, net benefit etc
	
*********************************************	
* ADDITIONAL THRESHOLD BASED MEASURES ADDED *
*********************************************

	local threshold=0.1
	gen positive=0
	replace positive = 1 if Predicted_P > `threshold'
	gen tp=1 if dead==1 & positive==1
	gen fp=1 if dead==0 & positive==1
	gen tn=1 if dead==0 & positive==0
	gen fn=1 if dead==1 & positive==0

	summ tp
	local TP = r(N)
	summ fp
	local FP = r(N)
	summ tn
	local TN = r(N)
	sum fn
	local FN = r(N)
	
	local outcome_prop = (`FN' + `TP') /_N

	summ dead
	local outcome_p = r(mean)
	
	
	* recall (sensitivity)
	
	local recall = `TP'/(`TP'+`FN')
    local se_recall=sqrt((`recall'*(1-`recall'))/(`TP'+`FN'))
	
	* precision
	local precision = `TP'/(`TP'+`FP')
	local se_precision=sqrt((`precision'*(1-`precision'))/(`TP'+`FP'))
	
	* specificity
	local spec=`TN'/(`TN'+`FP')
	local se_spec=sqrt((`spec'*(1-`spec'))/(`TN'+`FP'))
	
	* accuracy
	local accuracy=(`TP'+`TN')/_N
	local se_accuracy=sqrt((`accuracy'*(1-`accuracy'))/_N)
	
	* f1 score
	local f1 = 2/(1/`precision' + 1/`recall')

	local cov_1=(`precision'*(1-`precision')*(1-`recall'))/`outcome_prop'
	local cov_2=(`precision'*`spec'*(1-`precision'))/(1-`outcome_prop')
	local cov_p_r=(`cov_1'+`cov_2')/_N

	local se_f1= sqrt(4 * (`recall'^4*`se_precision'^2 + 2*`precision'^2*`recall'^2*`cov_p_r' + `precision'^4*`se_recall'^2) / (`precision'+`recall')^4)

	*NPV
	local npv=`TN'/(`TN'+`FN')
	local se_npv=sqrt((`npv'*(1-`npv'))/(_N*((`spec'*(1-`outcome_prop'))+(`outcome_prop'*(1-`recall')))))
	
	
	
**************************************	
	
	* store simulation ID and the obtained results
	local id = `i'
	post `testsize' (`id') (`slope')  (`se_slope') (`events') (`recall') (`se_recall') (`precision') (`se_precision') (`f1') (`se_f1') (`spec') (`se_spec') (`accuracy') (`se_accuracy') (`npv') (`se_npv')
	* drop variables ready for next loop
	drop LP skewnormal dead stime censtime Predicted_P cloglog_predicted positive tp tn fp fn
	* add a counter to help the user know simulation progress
	local gap = .
	nois disp `id' `gap' _continue
 }
 postclose `testsize'
 
 
* summarise results and calculate confidence intervals
 use `testfile', replace

 qui {
foreach measure in recall precision spec accuracy f1 npv {
summ `measure'
local `measure' = r(mean)
summ se_`measure'
local se_`measure' = r(mean)
 } 
 }
  
di "Sensitivity= " `recall' " (" `recall'-1.96*`se_recall' ", "  `recall'+1.96*`se_recall' ")"
di "PPV= " `precision' " (" `precision'-1.96*`se_precision' ", "  `precision'+1.96*`se_precision' ")"
di "Specificity= " `spec' " (" `spec'-1.96*`se_spec' ", "  `spec'+1.96*`se_spec' ")"
di "Accuracy= " `accuracy' " (" `accuracy'-1.96*`se_accuracy' ", "  `accuracy'+1.96*`se_accuracy' ")"
di "NPV= " `npv' " (" `npv'-1.96*`se_npv' ", "  `npv'+1.96*`se_npv' ")"
di "F1-score= " `f1' " (" `f1'-1.96*`se_f1' ", "  `f1'+1.96*`se_f1' ")"


  
  *** END ***

