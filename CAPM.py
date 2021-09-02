# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:23:01 2021

@author: byron
"""

import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from functools import reduce
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
from matplotlib.pyplot import figure
from matplotlib.font_manager import FontProperties

DATA = pd.read_excel("D.xlsx", parse_dates=True)
DATA.columns = ["PERMNO","Dates","Returns"]
DATA["Returns"] = DATA["Returns"]*100

FACTORS = pd.read_excel("FF.xlsx", parse_dates=True)
FACTORS.columns = ["Dates","SMB","HML","Momentum"]

INDEX = pd.read_excel("Index.xlsx", parse_dates = True)
INDEX.columns = ["Dates","S&P500"]
INDEX["S&P500"] = INDEX["S&P500"]*100

RF = pd.read_excel("RF.xls", parse_dates=True, skiprows = 10)
RF.columns = ["Dates","RF"]
RF.index = RF.Dates
RF = RF.drop(columns = ["Dates"])

FACTORS.index = INDEX.index
MERGE = pd.merge(FACTORS, INDEX)
MERGE.index = MERGE.Dates
MERGE = MERGE.drop(columns = ["Dates"])

Merge = MERGE.join(RF)

#OxfordTracker
OXFORD = pd.read_csv("OX.csv",parse_dates=True)
Usa = OXFORD[OXFORD.CountryCode == "USA"]
Usa.reset_index(inplace=True)
Usa = Usa[:596].drop(columns = 'index')
DATE = pd.to_datetime(Usa.Date, format = '%Y%m%d')
Usa.index = pd.to_datetime(DATE)
Usa = Usa.drop(columns=["Date"])
Usa = Usa.loc[Usa.index < "1-1-2021"]
Usa = Usa["StringencyIndex"]
Usa = Usa.resample("1M").mean()

#Basic Materials:
BHP = DATA[DATA.PERMNO == 75039]
SHW = DATA[DATA.PERMNO == 36468]
APD = DATA[DATA.PERMNO == 28222]
BMF = [BHP, SHW, APD]
BasicMaterials = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'], how = 'outer'), BMF)
BasicMaterials["PR"] = 0.33*BasicMaterials["Returns_x"] + 0.33*BasicMaterials["Returns_y"] + 0.33*BasicMaterials["Returns"]

#Change the index of the dataframes to Dates and drop some columns
BasicMaterials.index = Merge.index
BasicMaterials = BasicMaterials.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_BM = BasicMaterials["ER"] = BasicMaterials["PR"] - Merge["RF"]
X_BM = BasicMaterials["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_BM = sm.add_constant(X_BM)
Z_BM.columns = ["alpha","beta"]
MODEL_BM = sm.OLS(Y_BM,Z_BM)
RESULTS_BM = MODEL_BM.fit()
print(RESULTS_BM.summary())

#Acquire beta and alpha from the regression
BETA_BM = RESULTS_BM.params.beta
ALPHA_BM = RESULTS_BM.params.alpha

#Find the estimated CAPM returns
CAPM_R_BM = ALPHA_BM + BETA_BM*(X_BM)
#CAPM_Er = merge["Returns"] - CAPM_Returns

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_BM = BasicMaterials.loc[BasicMaterials.index < "31-12-2019"]
Y_RED_BM = REDUCED_BM["ER"]
X_RED_BM = REDUCED_BM["MRP"]
Z_RED_BM = sm.add_constant(X_RED_BM)
Z_RED_BM.columns = ["alpha","beta"]
MODEL_RED_BM = sm.OLS(Y_RED_BM,Z_RED_BM)
RESULTS_RED_BM = MODEL_RED_BM.fit()
print(RESULTS_RED_BM.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_BM = RESULTS_RED_BM.params.alpha
BETA_RED_BM = RESULTS_RED_BM.params.beta

#Create covid database and estimate market premium
COVID_BM = BasicMaterials.loc[BasicMaterials.index > "31-12-2019"]
X_COVID_BM = COVID_BM["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_BM = ALPHA_RED_BM + BETA_RED_BM*(X_COVID_BM)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_BM = CAPM_R_BM[CAPM_R_BM.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_BM = CAPM_R_RED_BM - CAPM_R_COVID_BM
Abnormal_sum_BM = sum(ABNORMAL_BM)

#Communication Services:
DIS = DATA[DATA.PERMNO == 26403] 
ATVI = DATA[DATA.PERMNO == 79678]
VOD = DATA[DATA.PERMNO == 75418]
CSF = [DIS, ATVI, VOD]
CommunicationServices = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'], how = 'outer'), CSF)
CommunicationServices["PR"] = 0.33*CommunicationServices["Returns_x"] + 0.33*CommunicationServices["Returns_y"] + 0.33*CommunicationServices["Returns"]

#Change the index of the dataframes to Dates and drop some columns
CommunicationServices.index = Merge.index
CommunicationServices = CommunicationServices.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_CS = CommunicationServices["ER"] = CommunicationServices["PR"] - Merge["RF"]
X_CS = CommunicationServices["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_CS = sm.add_constant(X_CS)
Z_CS.columns = ["alpha","beta"]
MODEL_CS = sm.OLS(Y_CS,Z_CS)
RESULTS_CS = MODEL_CS.fit()
print(RESULTS_CS.summary())

#Acquire beta and alpha from the regression
BETA_CS = RESULTS_CS.params.beta
ALPHA_CS = RESULTS_CS.params.alpha

#Find the estimated CAPM returns
CAPM_R_CS = ALPHA_CS + BETA_CS*(X_CS)

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_CS = CommunicationServices.loc[CommunicationServices.index < "31-12-2019"]
Y_RED_CS = REDUCED_CS["ER"]
X_RED_CS = REDUCED_CS["MRP"]
Z_RED_CS = sm.add_constant(X_RED_CS)
Z_RED_CS.columns = ["alpha","beta"]
MODEL_RED_CS = sm.OLS(Y_RED_CS,Z_RED_CS)
RESULTS_RED_CS = MODEL_RED_CS.fit()
print(RESULTS_RED_CS.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_CS = RESULTS_RED_CS.params.alpha
BETA_RED_CS = RESULTS_RED_CS.params.beta

#Create covid database and estimate market premium
COVID_CS = CommunicationServices.loc[CommunicationServices.index > "31-12-2019"]
X_COVID_CS = COVID_CS["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_CS = ALPHA_RED_CS + BETA_RED_CS*(X_COVID_CS)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_CS = CAPM_R_CS[CAPM_R_CS.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_CS = CAPM_R_RED_CS - CAPM_R_COVID_CS
Abnormal_sum_CS = sum(ABNORMAL_CS)

#Consumer Cyclical:
AMZN = DATA[DATA.PERMNO == 84788]
HD = DATA[DATA.PERMNO == 66181]
NKE = DATA[DATA.PERMNO == 57665]
CCF = [AMZN, HD, NKE]
ConsumerCyclical = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'],how = 'outer'), CCF)
ConsumerCyclical["PR"] = 0.33*ConsumerCyclical["Returns_x"] + 0.33*ConsumerCyclical["Returns_y"] + 0.33*ConsumerCyclical["Returns"]

#Change the index of the dataframes to Dates and drop some columns
ConsumerCyclical.index = Merge.index
ConsumerCyclical = ConsumerCyclical.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_CC = ConsumerCyclical["ER"] = ConsumerCyclical["PR"] - Merge["RF"]
X_CC = ConsumerCyclical["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_CC = sm.add_constant(X_CC)
Z_CC.columns = ["alpha","beta"]
MODEL_CC = sm.OLS(Y_CC,Z_CC)
RESULTS_CC = MODEL_CC.fit()
print(RESULTS_CC.summary())

#Acquire beta and alpha from the regression
BETA_CC = RESULTS_CC.params.beta
ALPHA_CC = RESULTS_CC.params.alpha

#Find the estimated CAPM returns
CAPM_R_CC = ALPHA_CC + BETA_CC*(X_CC)

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_CC = ConsumerCyclical.loc[ConsumerCyclical.index < "31-12-2019"]
Y_RED_CC = REDUCED_CC["ER"]
X_RED_CC = REDUCED_CC["MRP"]
Z_RED_CC = sm.add_constant(X_RED_CC)
Z_RED_CC.columns = ["alpha","beta"]
MODEL_RED_CC = sm.OLS(Y_RED_CC,Z_RED_CC)
RESULTS_RED_CC = MODEL_RED_CC.fit()
print(RESULTS_RED_CC.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_CC = RESULTS_RED_CC.params.alpha
BETA_RED_CC = RESULTS_RED_CC.params.beta

#Create covid database and estimate market premium
COVID_CC = ConsumerCyclical.loc[ConsumerCyclical.index > "31-12-2019"]
X_COVID_CC = COVID_CC["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_CC = ALPHA_RED_CC + BETA_RED_CC*(X_COVID_CC)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_CC = CAPM_R_CC[CAPM_R_CC.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_CC = CAPM_R_RED_CC - CAPM_R_COVID_CC
Abnormal_sum_CC = sum(ABNORMAL_CC)

#Consumer Defensive:
WMT = DATA[DATA.PERMNO == 55976]
PG = DATA[DATA.PERMNO == 18163]
KO = DATA[DATA.PERMNO == 11308]
CDF = [WMT, PG, KO]
ConsumerDefensive = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'],how = 'outer'), CDF)
ConsumerDefensive["PR"] = 0.33*ConsumerDefensive["Returns_x"] + 0.33*ConsumerDefensive["Returns_y"] + 0.33*ConsumerDefensive["Returns"]

#Change the index of the dataframes to Dates and drop some columns
ConsumerDefensive.index = Merge.index
ConsumerDefensive = ConsumerDefensive.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_CD = ConsumerDefensive["ER"] = ConsumerDefensive["PR"] - Merge["RF"]
X_CD = ConsumerDefensive["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_CD = sm.add_constant(X_CD)
Z_CD.columns = ["alpha","beta"]
MODEL_CD = sm.OLS(Y_CD,Z_CD)
RESULTS_CD = MODEL_CD.fit()
print(RESULTS_CD.summary())

#Acquire beta and alpha from the regression
BETA_CD = RESULTS_CD.params.beta
ALPHA_CD = RESULTS_CD.params.alpha

#Find the estimated CAPM returns
CAPM_R_CD = ALPHA_CD + BETA_CD*(X_CD)

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_CD = ConsumerDefensive.loc[ConsumerDefensive.index < "31-12-2019"]
Y_RED_CD = REDUCED_CD["ER"]
X_RED_CD = REDUCED_CD["MRP"]
Z_RED_CD = sm.add_constant(X_RED_CD)
Z_RED_CD.columns = ["alpha","beta"]
MODEL_RED_CD = sm.OLS(Y_RED_CD,Z_RED_CD)
RESULTS_RED_CD = MODEL_RED_CD.fit()
print(RESULTS_RED_CD.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_CD = RESULTS_RED_CD.params.alpha
BETA_RED_CD = RESULTS_RED_CD.params.beta

#Create covid database and estimate market premium
COVID_CD = ConsumerDefensive.loc[ConsumerDefensive.index > "31-12-2019"]
X_COVID_CD = COVID_CD["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_CD = ALPHA_RED_CD + BETA_RED_CD*(X_COVID_CD)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_CD = CAPM_R_CD[CAPM_R_CD.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_CD = CAPM_R_RED_CD - CAPM_R_COVID_CD
Abnormal_sum_CD = sum(ABNORMAL_CD)

#Energy:
XOM = DATA[DATA.PERMNO == 11850]
EPD = DATA[DATA.PERMNO == 86223]
TRP = DATA[DATA.PERMNO == 67774]
EF = [XOM, EPD, TRP]
Energy = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'],how = 'outer'), EF).dropna()
Energy["PR"] = 0.33*Energy["Returns_x"] + 0.33*Energy["Returns_y"] + 0.33*Energy["Returns"]

#Change the index of the dataframes to Dates and drop some columns
Energy.index = Merge.index
Energy = Energy.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_E = Energy["ER"] = Energy["PR"] - Merge["RF"]
X_E = Energy["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_E = sm.add_constant(X_E)
Z_E.columns = ["alpha","beta"]
MODEL_E = sm.OLS(Y_E,Z_E)
RESULTS_E = MODEL_E.fit()
print(RESULTS_E.summary())

#Acquire beta and alpha from the regression
BETA_E = RESULTS_E.params.beta
ALPHA_E = RESULTS_E.params.alpha

#Find the estimated CAPM returns
CAPM_R_E = ALPHA_E + BETA_E*(X_E)

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_E = Energy.loc[Energy.index < "31-12-2019"]
Y_RED_E = REDUCED_E["ER"]
X_RED_E = REDUCED_E["MRP"]
Z_RED_E = sm.add_constant(X_RED_E)
Z_RED_E.columns = ["alpha","beta"]
MODEL_RED_E = sm.OLS(Y_RED_E,Z_RED_E)
RESULTS_RED_E = MODEL_RED_E.fit()
print(RESULTS_RED_E.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_E = RESULTS_RED_E.params.alpha
BETA_RED_E = RESULTS_RED_E.params.beta

#Create covid database and estimate market premium
COVID_E = Energy.loc[Energy.index > "31-12-2019"]
X_COVID_E = COVID_E["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_E = ALPHA_RED_E + BETA_RED_E*(X_COVID_E)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_E = CAPM_R_E[CAPM_R_E.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_E = CAPM_R_RED_E - CAPM_R_COVID_E
Abnormal_sum_E = sum(ABNORMAL_E)

#Financial Services:
JPM = DATA[DATA.PERMNO == 47896]
BAC = DATA[DATA.PERMNO == 59408]
WFC = DATA[DATA.PERMNO == 38703]
FSF = [JPM, BAC, WFC]
FinancialServices = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'],how = 'outer'), FSF)
FinancialServices["PR"] = 0.33*FinancialServices["Returns_x"] + 0.33*FinancialServices["Returns_y"] + 0.33*FinancialServices["Returns"]

#Change the index of the dataframes to Dates and drop some columns
FinancialServices.index = Merge.index
FinancialServices = FinancialServices.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_FS = FinancialServices["ER"] = FinancialServices["PR"] - Merge["RF"]
X_FS = FinancialServices["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_FS = sm.add_constant(X_FS)
Z_FS.columns = ["alpha","beta"]
MODEL_FS = sm.OLS(Y_FS,Z_FS)
RESULTS_FS = MODEL_FS.fit()
print(RESULTS_FS.summary())

#Acquire beta and alpha from the regression
BETA_FS = RESULTS_FS.params.beta
ALPHA_FS = RESULTS_FS.params.alpha

#Find the estimated CAPM returns
CAPM_R_FS = ALPHA_FS + BETA_FS*(X_FS)

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_FS = FinancialServices.loc[FinancialServices.index < "31-12-2019"]
Y_RED_FS = REDUCED_FS["ER"]
X_RED_FS = REDUCED_FS["MRP"]
Z_RED_FS = sm.add_constant(X_RED_FS)
Z_RED_FS.columns = ["alpha","beta"]
MODEL_RED_FS = sm.OLS(Y_RED_FS,Z_RED_FS)
RESULTS_RED_FS = MODEL_RED_FS.fit()
print(RESULTS_RED_FS.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_FS = RESULTS_RED_FS.params.alpha
BETA_RED_FS = RESULTS_RED_FS.params.beta

#Create covid database and estimate market premium
COVID_FS = FinancialServices.loc[FinancialServices.index > "31-12-2019"]
X_COVID_FS = COVID_FS["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_FS = ALPHA_RED_FS + BETA_RED_FS*(X_COVID_FS)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_FS = CAPM_R_FS[CAPM_R_FS.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FS = CAPM_R_RED_FS - CAPM_R_COVID_FS
Abnormal_sum_FS = sum(ABNORMAL_FS)

#Healthcare:
JNJ = DATA[DATA.PERMNO == 22111]
UNH = DATA[DATA.PERMNO == 92655]
PFE = DATA[DATA.PERMNO == 21936]
HF = [JNJ, UNH, PFE]
Healthcare = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'],how = 'outer'), HF)
Healthcare["PR"] = 0.33*Healthcare["Returns_x"] + 0.33*Healthcare["Returns_y"] + 0.33*Healthcare["Returns"]

#Change the index of the dataframes to Dates and drop some columns
Healthcare.index = Merge.index
Healthcare = Healthcare.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_H = Healthcare["ER"] = Healthcare["PR"] - Merge["RF"]
X_H = Healthcare["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_H = sm.add_constant(X_H)
Z_H.columns = ["alpha","beta"]
MODEL_H = sm.OLS(Y_H,Z_H)
RESULTS_H = MODEL_H.fit()
print(RESULTS_H.summary())

#Acquire beta and alpha from the regression
BETA_H = RESULTS_H.params.beta
ALPHA_H = RESULTS_H.params.alpha

#Find the estimated CAPM returns
CAPM_R_H = ALPHA_H + BETA_H*(X_H)

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_H = Healthcare.loc[Healthcare.index < "31-12-2019"]
Y_RED_H = REDUCED_H["ER"]
X_RED_H = REDUCED_H["MRP"]
Z_RED_H = sm.add_constant(X_RED_H)
Z_RED_H.columns = ["alpha","beta"]
MODEL_RED_H = sm.OLS(Y_RED_H,Z_RED_H)
RESULTS_RED_H = MODEL_RED_H.fit()
print(RESULTS_RED_H.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_H = RESULTS_RED_H.params.alpha
BETA_RED_H = RESULTS_RED_H.params.beta

#Create covid database and estimate market premium
COVID_H = Healthcare.loc[Healthcare.index > "31-12-2019"]
X_COVID_H = COVID_H["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_H = ALPHA_RED_H + BETA_RED_H*(X_COVID_H)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_H = CAPM_R_H[CAPM_R_H.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_H = CAPM_R_RED_H - CAPM_R_COVID_H
Abnormal_sum_H = sum(ABNORMAL_H)

#Industrials:
UPS = DATA[DATA.PERMNO == 87447]
HON = DATA[DATA.PERMNO == 10145]
UNP = DATA[DATA.PERMNO == 48725]
IF = [UPS, HON, UNP]
Industrials = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'],how = 'outer'), IF)
Industrials["PR"] = 0.33*Industrials["Returns_x"] + 0.33*Industrials["Returns_y"] + 0.33*Industrials["Returns"]

#Change the index of the dataframes to Dates and drop some columns
Industrials.index = Merge.index
Industrials = Industrials.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_I = Industrials["ER"] = Industrials["PR"] - Merge["RF"]
X_I = Industrials["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_I = sm.add_constant(X_I)
Z_I.columns = ["alpha","beta"]
MODEL_I = sm.OLS(Y_I,Z_I)
RESULTS_I = MODEL_I.fit()
print(RESULTS_I.summary())

#Acquire beta and alpha from the regression
BETA_I = RESULTS_I.params.beta
ALPHA_I = RESULTS_I.params.alpha

#Find the estimated CAPM returns
CAPM_R_I = ALPHA_I + BETA_I*(X_I)

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_I = Industrials.loc[Industrials.index < "31-12-2019"]
Y_RED_I = REDUCED_I["ER"]
X_RED_I = REDUCED_I["MRP"]
Z_RED_I = sm.add_constant(X_RED_I)
Z_RED_I.columns = ["alpha","beta"]
MODEL_RED_I = sm.OLS(Y_RED_I,Z_RED_I)
RESULTS_RED_I = MODEL_RED_I.fit()
print(RESULTS_RED_I.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_I = RESULTS_RED_I.params.alpha
BETA_RED_I = RESULTS_RED_I.params.beta

#Create covid database and estimate market premium
COVID_I = Industrials.loc[Industrials.index > "31-12-2019"]
X_COVID_I = COVID_I["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_I = ALPHA_RED_I + BETA_RED_I*(X_COVID_I)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_I = CAPM_R_I[CAPM_R_I.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_I = CAPM_R_RED_I - CAPM_R_COVID_I
Abnormal_sum_I = sum(ABNORMAL_I)

#Real Estate:   
AMT = DATA[DATA.PERMNO == 86111]
CCI = DATA[DATA.PERMNO == 86339]
SPG = DATA[DATA.PERMNO == 80100]
REF = [AMT, CCI, SPG]
RealEstate = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'],how = 'outer'), REF)
RealEstate["PR"] = 0.33*RealEstate["Returns_x"] + 0.33*RealEstate["Returns_y"] + 0.33*RealEstate["Returns"]

#Change the index of the dataframes to Dates and drop some columns
RealEstate.index = Merge.index
RealEstate = RealEstate.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_RE = RealEstate["ER"] = RealEstate["PR"] - Merge["RF"]
X_RE = RealEstate["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_RE = sm.add_constant(X_RE)
Z_RE.columns = ["alpha","beta"]
MODEL_RE = sm.OLS(Y_RE,Z_RE)
RESULTS_RE = MODEL_RE.fit()
print(RESULTS_RE.summary())

#Acquire beta and alpha from the regression
BETA_RE = RESULTS_RE.params.beta
ALPHA_RE = RESULTS_RE.params.alpha

#Find the estimated CAPM returns
CAPM_R_RE = ALPHA_RE + BETA_RE*(X_RE)

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_RE = RealEstate.loc[RealEstate.index < "31-12-2019"]
Y_RED_RE = REDUCED_RE["ER"]
X_RED_RE = REDUCED_RE["MRP"]
Z_RED_RE = sm.add_constant(X_RED_RE)
Z_RED_RE.columns = ["alpha","beta"]
MODEL_RED_RE = sm.OLS(Y_RED_RE,Z_RED_RE)
RESULTS_RED_RE = MODEL_RED_RE.fit()
print(RESULTS_RED_RE.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_RE = RESULTS_RED_RE.params.alpha
BETA_RED_RE = RESULTS_RED_RE.params.beta

#Create covid database and estimate market premium
COVID_RE = RealEstate.loc[RealEstate.index > "31-12-2019"]
X_COVID_RE = COVID_RE["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_RE = ALPHA_RED_RE + BETA_RED_RE*(X_COVID_RE)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_RE = CAPM_R_RE[CAPM_R_RE.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_RE = CAPM_R_RED_RE - CAPM_R_COVID_RE
Abnormal_sum_RE = sum(ABNORMAL_RE)

#Tech:
AAPL = DATA[DATA.PERMNO == 14593]
MSFT = DATA[DATA.PERMNO == 10107]
TSM = DATA[DATA.PERMNO == 85442]
TF = [AAPL, MSFT, TSM]
Technology = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'],how = 'outer'), TF)
Technology["PR"] = 0.33*Technology["Returns_x"] + 0.33*Technology["Returns_y"] + 0.33*Technology["Returns"]

#Change the index of the dataframes to Dates and drop some columns
Technology.index = Merge.index
Technology = Technology.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_T = Technology["ER"] = Technology["PR"] - Merge["RF"]
X_T = Technology["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_T = sm.add_constant(X_T)
Z_T.columns = ["alpha","beta"]
MODEL_T = sm.OLS(Y_T,Z_T)
RESULTS_T = MODEL_T.fit()
print(RESULTS_T.summary())

#Acquire beta and alpha from the regression
BETA_T = RESULTS_T.params.beta
ALPHA_T = RESULTS_T.params.alpha

#Find the estimated CAPM returns
CAPM_R_T = ALPHA_T + BETA_T*(X_T)

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_T = Technology.loc[Technology.index < "31-12-2019"]
Y_RED_T = REDUCED_T["ER"]
X_RED_T = REDUCED_T["MRP"]
Z_RED_T = sm.add_constant(X_RED_T)
Z_RED_T.columns = ["alpha","beta"]
MODEL_RED_T = sm.OLS(Y_RED_T,Z_RED_T)
RESULTS_RED_T = MODEL_RED_T.fit()
print(RESULTS_RED_T.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_T = RESULTS_RED_T.params.alpha
BETA_RED_T = RESULTS_RED_T.params.beta

#Create covid database and estimate market premium
COVID_T = Technology.loc[Technology.index > "31-12-2019"]
X_COVID_T = COVID_T["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_T = ALPHA_RED_T + BETA_RED_T*(X_COVID_T)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_T = CAPM_R_T[CAPM_R_T.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_T = CAPM_R_RED_T - CAPM_R_COVID_T
Abnormal_sum_T = sum(ABNORMAL_T)

#Utilities:
DUK = DATA[DATA.PERMNO == 27959]
SO = DATA[DATA.PERMNO == 18411] 
NGG = DATA[DATA.PERMNO == 87280]
UF = [DUK, SO, NGG]
Utilities = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'],how = 'outer'), UF)
Utilities["PR"] = 0.33*Utilities["Returns_x"] + 0.33*Utilities["Returns_y"] + 0.33*Utilities["Returns"]

#Change the index of the dataframes to Dates and drop some columns
Utilities.index = Merge.index
Utilities = Utilities.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_U = Utilities["ER"] = Utilities["PR"] - Merge["RF"]
X_U = Utilities["MRP"] =  Merge["S&P500"] -  Merge["RF"]

#Add constant to market premium to act as intercept and run OLS regression
Z_U = sm.add_constant(X_U)
Z_U.columns = ["alpha","beta"]
MODEL_U = sm.OLS(Y_U,Z_U)
RESULTS_U = MODEL_U.fit()
print(RESULTS_U.summary())

#Acquire beta and alpha from the regression
BETA_U = RESULTS_U.params.beta
ALPHA_U = RESULTS_U.params.alpha

#Find the estimated CAPM returns
CAPM_R_U = ALPHA_U + BETA_U*(X_U)

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_U = Utilities.loc[Utilities.index < "31-12-2019"]
Y_RED_U = REDUCED_U["ER"]
X_RED_U = REDUCED_U["MRP"]
Z_RED_U = sm.add_constant(X_RED_U)
Z_RED_U.columns = ["alpha","beta"]
MODEL_RED_U = sm.OLS(Y_RED_U,Z_RED_U)
RESULTS_RED_U = MODEL_RED_U.fit()
print(RESULTS_RED_U.summary())

#Acquire the pre-covid alpha and beta
ALPHA_RED_U = RESULTS_RED_U.params.alpha
BETA_RED_U = RESULTS_RED_U.params.beta

#Create covid database and estimate market premium
COVID_U = Utilities.loc[Utilities.index > "31-12-2019"]
X_COVID_U = COVID_U["MRP"]

#Estimate the CAPM returns for the Covid database
CAPM_R_COVID_U = ALPHA_RED_U + BETA_RED_U*(X_COVID_U)

#Acquire the Actual CAPM returns for covid era that were estimated previously
CAPM_R_RED_U = CAPM_R_U[CAPM_R_U.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_U = CAPM_R_RED_U - CAPM_R_COVID_U
Abnormal_sum_U = sum(ABNORMAL_U)

#Create Dataframe containing all the abnormal returns
DICT = [ABNORMAL_BM,ABNORMAL_CS,ABNORMAL_CC,ABNORMAL_CD,ABNORMAL_E,ABNORMAL_FS,ABNORMAL_H,ABNORMAL_I,ABNORMAL_RE,ABNORMAL_T,ABNORMAL_U]
Di = pd.concat(DICT, axis=1)
x ={'Abnormal Returns' : [Abnormal_sum_BM,Abnormal_sum_CS,Abnormal_sum_CC,Abnormal_sum_CD,Abnormal_sum_E,Abnormal_sum_FS,Abnormal_sum_H,Abnormal_sum_I,Abnormal_sum_RE,Abnormal_sum_T,Abnormal_sum_U], 'Sector' : ['Basic Materials','Communication Services','Consumer Cyclical','Consumer Defensive','Energy','Financial Services','Healthcare','Industrials','Real Estate','Technology','Utilities']}
Di.columns = columns = x['Sector']

#Plotting of Abnormal Returns
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')
figure(figsize=(14, 10), dpi=100)
BM, = plt.plot(ABNORMAL_BM)
CS, = plt.plot(ABNORMAL_CS)
CC, = plt.plot(ABNORMAL_CC)
CD, = plt.plot(ABNORMAL_CD)
EN, = plt.plot(ABNORMAL_E)
FS, = plt.plot(ABNORMAL_FS)
HE, = plt.plot(ABNORMAL_H)
IN, = plt.plot(ABNORMAL_I)
RE, = plt.plot(ABNORMAL_RE)
TE, = plt.plot(ABNORMAL_T)
UT, = plt.plot(ABNORMAL_U)
plt.title("Abnormal Returns", fontproperties=font,fontsize='large')
plt.xlabel("Date", fontproperties=font,fontsize='large')
plt.ylabel("Precentage abnormal return", fontproperties=font,fontsize='large')
plt.legend([BM,CS,CC,CD,EN,FS,HE,IN,RE,TE,UT],['Basic Materials','Communication Services','Consumer Cyclical','Consumer Defensive','Energy','Financial Services','Healthcare','Industrials','Real Estate','Technology','Utilities'])
plt.show()

#Plotting of monthly Abnormal Returns
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')
DI_C = Di.resample("1M").mean()
figure(figsize=(14, 10), dpi=100)
BM_M, = plt.plot(DI_C["Basic Materials"])
CS_M, = plt.plot(DI_C["Communication Services"])
CC_M, = plt.plot(DI_C["Consumer Cyclical"])
CD_M, = plt.plot(DI_C["Consumer Defensive"])
EN_M, = plt.plot(DI_C["Energy"])
FS_M, = plt.plot(DI_C["Financial Services"])
HE_M, = plt.plot(DI_C["Healthcare"])
IN_M, = plt.plot(DI_C["Industrials"])
RE_M, = plt.plot(DI_C["Real Estate"])
TE_M, = plt.plot(DI_C["Technology"])
UT_M, = plt.plot(DI_C["Utilities"], linestyle = '--')
plt.legend([BM_M,CS_M,CC_M,CD_M,EN_M,FS_M,HE_M,IN_M,RE_M,TE_M,UT_M],['Basic Materials','Communication Services','Consumer Cyclical','Consumer Defensive','Energy','Financial Services','Healthcare','Industrials','Real Estate','Technology','Utilities'])
plt.title("Abnormal Returns", fontproperties=font,fontsize='large')
plt.xlabel("Date", fontproperties=font,fontsize='large')
plt.ylabel("Precentage abnormal return", fontproperties=font,fontsize='large')
plt.show()

#Plotting of cumulated Abnormal Returns
X1 = pd.DataFrame(x)
X1 = X1.sort_values(by=['Abnormal Returns'])
X1.reset_index(inplace=True)
X1 = X1.drop(columns = "index")
X1['colors'] = ['red' if x < 0 else 'green' for x in X1['Abnormal Returns']]
plt.figure(figsize=(14,10), dpi= 100)
plt.hlines(y = X1.index, xmin=0, xmax = X1['Abnormal Returns'], color=X1.colors, alpha=0.5, linewidth=5)
plt.ylabel("Sector", fontproperties=font, fontsize= 22)
plt.xlabel("Abnormal Returns", fontproperties=font, fontsize= 22)
plt.yticks(X1.index, X1.Sector, fontsize=12)
plt.title('Cumulated Abnormal Returns', fontproperties=font, fontsize= 26)
plt.grid(linestyle='--', alpha=0.5)
plt.show()

Mean = X1.mean()
Var = X1.var()

fig, axs = plt.subplots(2, figsize=(22, 16), dpi=200)

#Boxplots
axs[0].boxplot(Di, vert = 0, meanline=True, showmeans=True, showfliers=True)
axs[0].set_yticklabels(Di.columns, fontsize= 12)
axs[0].set_title("Boxplots of Abnormal Returns", fontproperties=font, fontsize= 26)
axs[0].set_ylabel("Sectors", fontproperties=font, fontsize= 22)
axs[0].set_xlabel("Abnormal Returns", fontproperties=font, fontsize= 22)
axs[0].grid(linestyle='--', alpha=0.5)
#plt.show()

#Volatility Estimates
volatility = Di.std()*252**.5
volatility = pd.DataFrame(volatility)
volatility.columns = ['V']
volatility = volatility.sort_values(by=['V'])
volatility['colors'] = ['red' if x < 0 else 'green' for x in volatility['V']]
# plt.figure(figsize=(14,10), dpi= 100)
axs[1].hlines(y = volatility.index, xmin=0, xmax = volatility['V'], color=volatility.colors, alpha=0.5, linewidth=5)
axs[1].set_xlabel("Volatility", fontproperties=font, fontsize= 22)
axs[1].set_ylabel("Sectors", fontproperties=font, fontsize= 22)
axs[1].set_title('Volatility of Abnormal Returns', fontproperties=font, fontsize= 26)
#axs[1].yticks(X1.index, X1.Sector, fontsize=12)
axs[1].grid(linestyle='--', alpha=0.5)
plt.show()

#Correlation
Di_M = Di.resample('M').mean()
Di_M = Di_M.join(Usa)
#Basic Materials
corr1_BM, _ = pearsonr(Di_M["Basic Materials"], Di_M["StringencyIndex"])
corr2_BM, _ = spearmanr(Di_M["Basic Materials"], Di_M["StringencyIndex"])
#Communication Services
corr1_CS, _ = pearsonr(Di_M["Communication Services"], Di_M["StringencyIndex"])
corr2_CS, _ = spearmanr(Di_M["Communication Services"], Di_M["StringencyIndex"])
#Consumer Cyclical
corr1_CC, _ = pearsonr(Di_M["Consumer Cyclical"], Di_M["StringencyIndex"])
corr2_CC, _ = spearmanr(Di_M["Consumer Cyclical"], Di_M["StringencyIndex"])
#Consumer Defensive
corr1_CD, _ = pearsonr(Di_M["Consumer Defensive"], Di_M["StringencyIndex"])
corr2_CD, _ = spearmanr(Di_M["Consumer Defensive"], Di_M["StringencyIndex"])
#Energy
corr1_E, _ = pearsonr(Di_M["Energy"], Di_M["StringencyIndex"])
corr2_E, _ = spearmanr(Di_M["Energy"], Di_M["StringencyIndex"])
#Financial Services
corr1_FS, _ = pearsonr(Di_M["Financial Services"], Di_M["StringencyIndex"])
corr2_FS, _ = spearmanr(Di_M["Financial Services"], Di_M["StringencyIndex"])
#Healthcare
corr1_H, _ = pearsonr(Di_M["Healthcare"], Di_M["StringencyIndex"])
corr2_H, _ = spearmanr(Di_M["Healthcare"], Di_M["StringencyIndex"])
#Industrials
corr1_I, _ = pearsonr(Di_M["Industrials"], Di_M["StringencyIndex"])
corr2_I, _ = spearmanr(Di_M["Industrials"], Di_M["StringencyIndex"])
#Real Estate
corr1_RE, _ = pearsonr(Di_M["Real Estate"], Di_M["StringencyIndex"])
corr2_RE, _ = spearmanr(Di_M["Real Estate"], Di_M["StringencyIndex"]) 
#Technology
corr1_T, _ = pearsonr(Di_M["Technology"], Di_M["StringencyIndex"])
corr2_T, _ = spearmanr(Di_M["Technology"], Di_M["StringencyIndex"])
#Utilities
corr1_U, _ = pearsonr(Di_M["Utilities"], Di_M["StringencyIndex"])
corr2_U, _ = spearmanr(Di_M["Utilities"], Di_M["StringencyIndex"])

#Correlation plots
CORR_PEARSON = [corr1_BM,corr1_CS,corr1_CC,corr1_CD,corr1_E,corr1_FS,corr1_H,corr1_I,corr1_RE,corr1_T,corr1_U]
CORR_SPEARMAN = [corr2_BM,corr2_CS,corr2_CC,corr2_CD,corr2_E,corr2_FS,corr2_H,corr2_I,corr2_RE,corr2_T,corr2_U]
X_AXIS = np.arange(len(x['Sector']))
fig, ax = plt.subplots(figsize=(30, 10), dpi=100)
plt.bar(X_AXIS - 0.2, CORR_PEARSON, 0.4, label = 'Pearson')
plt.bar(X_AXIS + 0.2, CORR_SPEARMAN, 0.4, label = 'Spearman')
plt.xticks(X_AXIS, x['Sector'])
plt.xlabel("Sectors",fontproperties=font, fontsize= 18)
plt.ylabel("Correlations",fontproperties=font, fontsize= 18)
plt.title("Correlation with Oxcford Government Response Tracker",fontproperties=font, fontsize= 26)
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.show()