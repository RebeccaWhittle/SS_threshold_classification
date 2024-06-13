# import necessary modules
import itertools as it
import numpy as np
import pandas as pd
import math
from scipy.stats import norm, beta
from scipy.special import logit

# simulate data based on known distribution and chosen probability threshold (p=0.1)
np.random.seed(1234)
N = 1000000
a, b = 1.33, 1.75
threshold = 0.1
df = pd.DataFrame(data= {'P': beta.rvs(a, b, size=N)})
df['outcome'] = np.random.binomial(1, df['P'], N)
df['positive'] = pd.Series(0, index=df.index).mask(df['P']>threshold, 1)

# calculate sensitivity, specifiity and outcome proportion of simulated data
sens = df.positive[df.outcome==1].mean()
spec = 1 - df.positive[df.outcome==0].mean()
outcome_prop = df['outcome'].mean()

# calcaulte true positivies, negatives etc of simulated data
df.loc[(df['outcome']==1) & (df['positive']==1), 'tp'] = 1
df.loc[(df['outcome']==0) & (df['positive']==1), 'fp'] = 1
df.loc[(df['outcome']==0) & (df['positive']==0), 'tn'] = 1
df.loc[(df['outcome']==1) & (df['positive']==0), 'fn'] = 1
TP = len(df[df.tp==1])
FP = len(df[df.fp==1])
TN = len(df[df.tn==1])
FN = len(df[df.fn==1])

## calculate threshold-based eprformance measure values from simulated data (i.e. assumed true values)

# accuracy
accuracy=(TP+TN)/N
# specificity
spec=TN/(TN+FP)
# sensitivity / recall
sens = TP/(TP+FN)
# NPV
npv=TN/(TN+FN)
# precision / PPV
ppv = TP / (TP + FP)
# F1 score
recall = sens
precision = ppv
F1 = 2/(1/precision + 1/recall)


### SAMPLE SIZE CALCULATIONS

# target standard error (to target a CI width of 0.1 for all measures)
se = 0.0255

# set assumed true values of performance measures based on calculated values from simulated data
outcome_prop = round(outcome_prop, 4)
accuracy = round(accuracy, 4)
spec = round(spec, 4)
sens, recall = round(sens, 4), round(sens, 4)
ppv, precision = round(ppv, 4), round(ppv, 4)
npv = round(npv, 4)
f1 = round(F1, 4)

# calculate sample size for target standard error
SS_accuracy = (accuracy * (1-accuracy)) / se**2
SS_spec = (spec * (1-spec)) / (se**2 * (1-outcome_prop))
SS_sens = (sens * (1-sens)) / (outcome_prop * se**2)
SS_ppv = ((ppv**2) * (1-ppv)) / (se**2 * outcome_prop * sens)
SS_npv = (npv * (1-npv)) / (se**2 * (spec * (1-outcome_prop) + outcome_prop * (1-sens)))
cov_1 = (precision * (1-precision) * (1-recall)) / outcome_prop
cov_2 = (precision * spec * (1-precision)) / (1-outcome_prop)
SS_f1 = (2 * (precision**2) * (recall**2) * (cov_1 + cov_2)) / ((((se**2) * ((precision + recall)**4)) / 4) - ((precision**4) * (se**2)) - ((recall**4) * (se**2)))

# print SS calculations:
print("Minimum required sample size (events):")
print("Accuracy: " + str(math.ceil(SS_accuracy)) + " (" + str(math.ceil(SS_accuracy*outcome_prop)) + ")")
print("Specificity: " + str(math.ceil(SS_spec)) + " (" + str(math.ceil(SS_spec*outcome_prop)) + ")")
print("Sensitvity/Recall: " + str(math.ceil(SS_sens)) + " (" + str(math.ceil(SS_sens*outcome_prop)) + ")")
print("PPV/Precision: " + str(math.ceil(SS_ppv)) + " (" + str(math.ceil(SS_ppv*outcome_prop)) + ")")
print("NPV: " + str(math.ceil(SS_npv)) + " (" + str(math.ceil(SS_npv*outcome_prop)) + ")")
print("F1-Score: " + str(math.ceil(SS_f1)) + " (" + str(math.ceil(SS_f1*outcome_prop)) + ")")



### Calculate precision (CIs) of performance measures using minimum SS from Riley criteria

# first calculate minmum sample size using Riley criteria 

# Step (i): O/E calculation
## target CI width of 0.22 for O/E, so SE(lnOE) = 0.056
selnoe = 0.056
outcome_prop = 0.43
SS_OE = (1 - outcome_prop) / (outcome_prop * selnoe**2)

# Step (ii): calibration slope
a, b = 1.33, 1.75
N = 1000000
np.random.seed(1234)
df2 = pd.DataFrame(data= {'P': beta.rvs(a, b, size=N)})
df2['LP'] = logit(df2['P'])
beta0 = 0
beta1 = 1
df2['Borenstein_00'] = np.exp(beta0 + (beta1 * df2['LP'])) / ((1 + np.exp(beta0 + (beta1 * df2['LP'])))**2)
df2['Borenstein_01'] = df2['LP'] * np.exp(beta0 + (beta1 * df2['LP'])) / ((1+ np.exp(beta0 + (beta1 * df2['LP'])))**2)
df2['Borenstein_11'] = df2['LP'] * df2['LP'] * np.exp(beta0 + (beta1 * df2['LP'])) / ((1 + np.exp(beta0 + (beta1 * df2['LP'])))**2)
I_00 = df2['Borenstein_00'].mean()
I_01 = df2['Borenstein_01'].mean()
I_11 = df2['Borenstein_11'].mean()
#target CI width of 0.3, correspondds to SE of 0.07653
seslope = 0.07653
SS_slope = (I_00 / (seslope*seslope * ((I_00*I_11) - (I_01*I_01))))

# Step (iii): C-statistic
# target CI width of 0.1, SE=0.02551
Cstat=0.77
df3 = pd.DataFrame(data = {'SS': df2.index+1})
df3['seCstatsq'] = Cstat * (1 - Cstat) * (1 + (((df3['SS'] / 2) - 1) * ((1 - Cstat) / (2 - Cstat))) + ((((df3['SS'] / 2) - 1) * Cstat) / (1 + Cstat))) / (df3['SS'] * df3['SS'] * outcome_prop * (1 - outcome_prop))
df3['seCstat'] = np.sqrt(df3['seCstatsq'])
df3['CIwidth'] = 2 * 1.96 * df3['seCstat']
df4 = df3.loc[df3.CIwidth<= 0.1000]
SS_cstat =df4['size'].min()

# Step (iv): Net benefit 
NB = (sens * outcome_prop) - ((1 - spec) * (1 - outcome_prop) * (threshold / (1 - threshold)))
sNB = NB / outcome_prop
w = ((1 - outcome_prop) / outcome_prop) * (threshold / (1 - threshold))
# target CI width 0.2, SE=0.051
sesNB = 0.051
SS_sNB = (1 / (sesNB**2)) * (((sens * (1 - sens)) / outcome_prop) + (w**2 * spec * (1 - spec) / (1 - outcome_prop)) + (w**2 * (1 - spec) * (1 - spec) / (outcome_prop * (1 - outcome_prop))))

# Minimum required sample size from Riley criteria:
minSS=math.ceil(max(SS_OE, SS_slope, SS_cstat, SS_sNB))
print("Minimum required sample size from Riley criteria: "+str(minSS))

## Sample size for classification performance measures based on riley min SS
# Accuracy
se_accuracy = math.sqrt((accuracy * (1 - accuracy)) / minSS)
accuracy_lci = accuracy - 1.96 * se_accuracy
accuracy_uci = accuracy + 1.96 * se_accuracy

# sensitivity/recall
se_sens = math.sqrt((sens * (1 - sens)) / (outcome_prop * minSS))
sens_lci = sens - 1.96 * se_sens
sens_uci = sens + 1.96 * se_sens

# Specificity
se_spec = math.sqrt((spec * (1 - spec)) / (minSS * (1 - outcome_prop)))
spec_lci = spec - 1.96 * se_spec
spec_uci = spec + 1.96 * se_spec

# PPV/precision
se_ppv = math.sqrt((ppv * (1 - ppv)) / (minSS * outcome_prop * sens * (1 / ppv)))
ppv_lci = ppv - 1.96 * se_ppv
ppv_uci = ppv + 1.96 * se_ppv

# NPV
se_npv = math.sqrt((npv * (1 - npv)) / (minSS * ((spec * (1 - outcome_prop)) + (outcome_prop * (1 - sens)))))
npv_lci = npv - 1.96 * se_npv
npv_uci = npv + 1.96 * se_npv

# F1-Score
recall = sens
precision = ppv
se_precision = se_ppv
se_recall = se_sens
f1 = 2 / (1 / precision + 1 / recall)
cov_1 = (precision * (1 - precision) * (1 - recall)) / outcome_prop
cov_2 = (precision * spec * (1 - precision)) / (1 - outcome_prop)
cov_p_r = (cov_1 + cov_2) / minSS
se_f1 = math.sqrt(4 * (((recall**4) * (se_precision**2)) + (2* (precision**2) * (recall**2) * cov_p_r) + ((precision**4) * (se_recall**2))) / ((precision + recall)**4))
f1_lci = f1 - 1.96 * se_f1
f1_uci = f1 + 1.96 * se_f1

# print performance measure values with CIs based on sample size from Riley criteria
print("Accuracy (95% CI) when sample size = " + str(minSS) + ": " + str(round(accuracy,3)) + " (" + str(round(accuracy_lci,3)) + ", " + str(round(accuracy_uci,3)) + ")")
print("Sensitivity/Recall (95% CI) when sample size = " + str(minSS) + ": " + str(round(sens,3)) + " (" + str(round(sens_lci,3)) + ", " + str(round(sens_uci,3)) + ")")
print("Specificity (95% CI) when sample size = " + str(minSS) + ": " + str(round(spec,3)) + " (" + str(round(spec_lci,3)) + ", " + str(round(spec_uci,3)) + ")")
print("PPV/Precision (95% CI) when sample size = " + str(minSS) + ": " + str(round(ppv,3)) + " (" + str(round(ppv_lci,3)) + ", " + str(round(ppv_uci,3)) + ")")
print("NPV (95% CI) when sample size = " + str(minSS) + ": " + str(round(npv,3)) + " (" + str(round(npv_lci,3)) + ", " + str(round(npv_uci,3)) + ")")
print("F1-Score (95% CI) when sample size = " + str(minSS) + ": " + str(round(f1,3)) + " (" + str(round(f1_lci,3)) + ", " + str(round(f1_uci,3)) + ")")





## SS using Agresti & Coull standard errors

# Point estimates
accuracy = (TP + TN) / (N)
spec = (TN) / (TN + FP)
sens = (TP) / (TP + FN)
ppv = (TP) / (TP + FP)
npv = (TN) / (TN + FN)

# Agresti & Coull point estimates
accuracy_ac=(TP+TN+2)/(N+4)
spec_ac=(TN+2)/(TN+FP+4)
sens_ac=(TP+2)/(TP+FN+4)
ppv_ac=(TP+2)/(TP+FP+4)
npv_ac=(TN+2)/(TN+FN+4)

# Set target CI width
ciw=0.1000

## ACCURACY
df_acc = pd.DataFrame(index=np.arange(10000000))
df_acc['SS'] = df_acc.index+1
df_acc['se_accuracy_ac'] = np.sqrt((accuracy_ac * (1 - accuracy_ac)) / df_acc['SS'])
df_acc['ciwidth'] = 2 * 1.96 * df_acc['se_accuracy_ac']
df_acc.loc[df_acc['ciwidth'] < ciw, 'keep'] = 1
SS_accuracy_ac = df_acc.groupby('keep').min().iloc[0,0]

## SPECIFICITY
df_spec = pd.DataFrame(index=np.arange(10000000))
df_spec['SS'] = df_spec.index+1
df_spec['se_spec_ac'] = np.sqrt((spec_ac * (1 - spec_ac)) / (df_spec['SS'] * (1-outcome_prop)))
df_spec['ciwidth'] = 2 * 1.96 * df_spec['se_spec_ac']
df_spec.loc[df_spec['ciwidth'] < ciw, 'keep'] = 1
SS_spec_ac = df_spec.groupby('keep').min().iloc[0,0]

## SENSITIVITY
df_sens = pd.DataFrame(index=np.arange(10000000))
df_sens['SS'] = df_sens.index+1
df_sens['se_sens_ac'] = np.sqrt((sens_ac * (1 - sens_ac)) / (df_sens['SS'] * outcome_prop))
df_sens['ciwidth'] = 2 * 1.96 * df_sens['se_sens_ac']
df_sens.loc[df_sens['ciwidth'] < ciw, 'keep'] = 1
SS_sens_ac = df_sens.groupby('keep').min().iloc[0,0]

## PPV/Precision
df_ppv = pd.DataFrame(index=np.arange(10000000))
df_ppv['SS'] = df_ppv.index+1
df_ppv['se_ppv_ac'] = np.sqrt((ppv_ac**2 * (1 - ppv_ac)) / (df_ppv['SS'] * outcome_prop * sens))
df_ppv['ciwidth'] = 2 * 1.96 * df_ppv['se_ppv_ac']
df_ppv.loc[df_ppv['ciwidth'] < ciw, 'keep'] = 1
SS_ppv_ac = df_ppv.groupby('keep').min().iloc[0,0]

## NPV
df_npv = pd.DataFrame(index=np.arange(10000000))
df_npv['SS'] = df_npv.index+1
df_npv['se_npv_ac'] = np.sqrt((npv_ac * (1 - npv_ac)) / (df_npv['SS'] * ((spec * (1 - outcome_prop)) + (outcome_prop * (1- sens)))))
df_npv['ciwidth'] = 2 * 1.96 * df_npv['se_npv_ac']
df_npv.loc[df_npv['ciwidth'] < ciw, 'keep'] = 1
SS_npv_ac = df_npv.groupby('keep').min().iloc[0,0]


print("Minimum required sample size (events) using Agresti&Coull standard errors:")
print("Accuracy: " + str(math.ceil(SS_accuracy_ac)) + " (" + str(math.ceil(SS_accuracy_ac*outcome_prop)) + ")")
print("Specificity: " + str(math.ceil(SS_spec_ac)) + " (" + str(math.ceil(SS_spec_ac*outcome_prop)) + ")")
print("Sensitvity/Recall: " + str(math.ceil(SS_sens_ac)) + " (" + str(math.ceil(SS_sens_ac*outcome_prop)) + ")")
print("PPV/Precision: " + str(math.ceil(SS_ppv_ac)) + " (" + str(math.ceil(SS_ppv_ac*outcome_prop)) + ")")
print("NPV: " + str(math.ceil(SS_npv_ac)) + " (" + str(math.ceil(SS_npv_ac*outcome_prop)) + ")")