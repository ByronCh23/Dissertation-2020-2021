# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 20:57:41 2021

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
Y_BM = Merge["ER"] = BasicMaterials["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_BM = Merge[["MRP","SMB","HML"]]
Z_BM = sm.add_constant(X_BM)
Z_BM.columns = ["alpha","beta1","beta2","beta3"]
MODEL_BM = sm.OLS(Y_BM,Z_BM)
RESULTS_BM = MODEL_BM.fit()
print(RESULTS_BM.summary())

#Acquire betas and alpha from the regression
ALPHA_BM = RESULTS_BM.params[0]
BETA1_BM = RESULTS_BM.params[1]
BETA2_BM = RESULTS_BM.params[2]
BETA3_BM = RESULTS_BM.params[3]

#Find the estimated FF returns
FF_RETURNS_BM = ALPHA_BM + BETA1_BM*Merge["MRP"] + BETA2_BM*Merge["SMB"] + BETA3_BM*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_BM = Merge.loc[Merge.index < "31-12-2019"]
Y_RED_BM = REDUCED_BM["ER"]
X_RED_BM = REDUCED_BM[["MRP","SMB","HML"]]
Z_RED_BM = sm.add_constant(X_RED_BM)
Z_RED_BM.columns = ["alpha","beta1","beta2","beta3"]
MODEL_RED_BM = sm.OLS(Y_RED_BM,Z_RED_BM)
RESULTS_RED_BM = MODEL_RED_BM.fit()
print(RESULTS_RED_BM.summary())

#Acquire the pre-covid alpha and betas
ALPHA_RED_BM = RESULTS_RED_BM.params.alpha
BETA1_RED_BM = RESULTS_RED_BM.params.beta1
BETA2_RED_BM = RESULTS_RED_BM.params.beta2
BETA3_RED_BM = RESULTS_RED_BM.params.beta3

#Create covid database 
COVID_BM = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_RETURNS_COVID_BM = ALPHA_RED_BM + BETA1_RED_BM*(COVID_BM["MRP"]) + BETA2_RED_BM*(COVID_BM["SMB"]) + BETA3_RED_BM*(COVID_BM["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_RETURNS_RED_BM = FF_RETURNS_BM[FF_RETURNS_BM.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_BM = FF_RETURNS_RED_BM - FF_RETURNS_COVID_BM
Abnormal_FF_sum_BM = sum(ABNORMAL_FF_BM)


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
Y_CS = Merge["ER"] = CommunicationServices["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_CS = Merge[["MRP","SMB","HML"]]
Z_CS = sm.add_constant(X_CS)
Z_CS.columns = ["alpha","beta1","beta2","beta3"]
MODEL_CS = sm.OLS(Y_CS,Z_CS)
RESULTS_CS = MODEL_CS.fit()
print(RESULTS_CS.summary())

#Acquire betas and alpha from the regression
ALPHA_CS = RESULTS_CS.params[0]
BETA1_CS = RESULTS_CS.params[1]
BETA2_CS = RESULTS_CS.params[2]
BETA3_CS = RESULTS_CS.params[3]

#Find the estimated FF returns
FF_RETURNS_CS = ALPHA_CS + BETA1_CS*Merge["MRP"] + BETA2_CS*Merge["SMB"] + BETA3_CS*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_CS = Merge.loc[Merge.index < "31-12-2019"]
Y_RED_CS = REDUCED_CS["ER"]
X_RED_CS = REDUCED_CS[["MRP","SMB","HML"]]
Z_RED_CS = sm.add_constant(X_RED_CS)
Z_RED_CS.columns = ["alpha","beta1","beta2","beta3"]
MODEL_RED_CS = sm.OLS(Y_RED_CS,Z_RED_CS)
RESULTS_RED_CS = MODEL_RED_CS.fit()
print(RESULTS_RED_CS.summary())

#Acquire the pre-covid alpha and betas
ALPHA_RED_CS = RESULTS_RED_CS.params.alpha
BETA1_RED_CS = RESULTS_RED_CS.params.beta1
BETA2_RED_CS = RESULTS_RED_CS.params.beta2
BETA3_RED_CS = RESULTS_RED_CS.params.beta3

#Create covid database 
COVID_CS = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_RETURNS_COVID_CS = ALPHA_RED_CS + BETA1_RED_CS*(COVID_CS["MRP"]) + BETA2_RED_CS*(COVID_CS["SMB"]) + BETA3_RED_CS*(COVID_CS["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_RETURNS_RED_CS = FF_RETURNS_CS[FF_RETURNS_CS.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_CS = FF_RETURNS_RED_CS - FF_RETURNS_COVID_CS
Abnormal_FF_sum_CS = sum(ABNORMAL_FF_CS)

#Consumer Cyclical:
AMZN = DATA[DATA.PERMNO == 84788]
HD = DATA[DATA.PERMNO == 66181]
NKE = DATA[DATA.PERMNO == 57665]
CCF = [AMZN, HD, NKE]
ConsumerCyclical = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'], how = 'outer'), CCF)
ConsumerCyclical["PR"] = 0.33*ConsumerCyclical["Returns_x"] + 0.33*ConsumerCyclical["Returns_y"] + 0.33*ConsumerCyclical["Returns"]

#Change the index of the dataframes to Dates and drop some columns
ConsumerCyclical.index = Merge.index
ConsumerCyclical = ConsumerCyclical.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_CC = Merge["ER"] = ConsumerCyclical["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_CC = Merge[["MRP","SMB","HML"]]
Z_CC = sm.add_constant(X_CC)
Z_CC.columns = ["alpha","beta1","beta2","beta3"]
MODEL_CC = sm.OLS(Y_CC,Z_CC)
RESULTS_CC = MODEL_CC.fit()
print(RESULTS_CC.summary())

#Acquire betas and alpha from the regression
ALPHA_CC = RESULTS_CC.params[0]
BETA1_CC = RESULTS_CC.params[1]
BETA2_CC = RESULTS_CC.params[2]
BETA3_CC = RESULTS_CC.params[3]

#Find the estimated FF returns
FF_RETURNS_CC = ALPHA_CC + BETA1_CC*Merge["MRP"] + BETA2_CC*Merge["SMB"] + BETA3_CC*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_CC = Merge.loc[Merge.index < "31-12-2019"]
Y_RED_CC = REDUCED_CC["ER"]
X_RED_CC = REDUCED_CC[["MRP","SMB","HML"]]
Z_RED_CC = sm.add_constant(X_RED_CC)
Z_RED_CC.columns = ["alpha","beta1","beta2","beta3"]
MODEL_RED_CC = sm.OLS(Y_RED_CC,Z_RED_CC)
RESULTS_RED_CC = MODEL_RED_CC.fit()
print(RESULTS_RED_CC.summary())

#Acquire the pre-covid alpha and betas
ALPHA_RED_CC = RESULTS_RED_CC.params.alpha
BETA1_RED_CC = RESULTS_RED_CC.params.beta1
BETA2_RED_CC = RESULTS_RED_CC.params.beta2
BETA3_RED_CC = RESULTS_RED_CC.params.beta3

#Create covid database 
COVID_CC = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_RETURNS_COVID_CC = ALPHA_RED_CC + BETA1_RED_CC*(COVID_CC["MRP"]) + BETA2_RED_CC*(COVID_CC["SMB"]) + BETA3_RED_CC*(COVID_CC["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_RETURNS_RED_CC = FF_RETURNS_CC[FF_RETURNS_CC.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_CC = FF_RETURNS_RED_CC - FF_RETURNS_COVID_CC
Abnormal_FF_sum_CC = sum(ABNORMAL_FF_CC)

#Consumer Defensive:
WMT = DATA[DATA.PERMNO == 55976]
PG = DATA[DATA.PERMNO == 18163]
KO = DATA[DATA.PERMNO == 11308]
CDF = [WMT, PG, KO]
ConsumerDefensive = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'], how = 'outer'), CDF)
ConsumerDefensive["PR"] = 0.33*ConsumerDefensive["Returns_x"] + 0.33*ConsumerDefensive["Returns_y"] + 0.33*ConsumerDefensive["Returns"]

#Change the index of the dataframes to Dates and drop some columns
ConsumerDefensive.index = Merge.index
ConsumerDefensive = ConsumerDefensive.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_CD = Merge["ER"] = ConsumerDefensive["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_CD = Merge[["MRP","SMB","HML"]]
Z_CD = sm.add_constant(X_CD)
Z_CD.columns = ["alpha","beta1","beta2","beta3"]
MODEL_CD = sm.OLS(Y_CD,Z_CD)
RESULTS_CD = MODEL_CD.fit()
print(RESULTS_CD.summary())

#Acquire betas and alpha from the regression
ALPHA_CD = RESULTS_CD.params[0]
BETA1_CD = RESULTS_CD.params[1]
BETA2_CD = RESULTS_CD.params[2]
BETA3_CD = RESULTS_CD.params[3]

#Find the estimated FF returns
FF_RETURNS_CD = ALPHA_CD + BETA1_CD*Merge["MRP"] + BETA2_CD*Merge["SMB"] + BETA3_CD*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_CD = Merge.loc[Merge.index < "31-12-2019"]
Y_RED_CD = REDUCED_CD["ER"]
X_RED_CD = REDUCED_CD[["MRP","SMB","HML"]]
Z_RED_CD = sm.add_constant(X_RED_CD)
Z_RED_CD.columns = ["alpha","beta1","beta2","beta3"]
MODEL_RED_CD = sm.OLS(Y_RED_CD,Z_RED_CD)
RESULTS_RED_CD = MODEL_RED_CD.fit()
print(RESULTS_RED_CD.summary())

#Acquire the pre-covid alpha and betas
ALPHA_RED_CD = RESULTS_RED_CD.params.alpha
BETA1_RED_CD = RESULTS_RED_CD.params.beta1
BETA2_RED_CD = RESULTS_RED_CD.params.beta2
BETA3_RED_CD = RESULTS_RED_CD.params.beta3

#Create covid database 
COVID_CD = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_RETURNS_COVID_CD = ALPHA_RED_CD + BETA1_RED_CD*(COVID_CD["MRP"]) + BETA2_RED_CD*(COVID_CD["SMB"]) + BETA3_RED_CD*(COVID_CD["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_RETURNS_RED_CD = FF_RETURNS_CD[FF_RETURNS_CD.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_CD = FF_RETURNS_RED_CD - FF_RETURNS_COVID_CD
Abnormal_FF_sum_CD = sum(ABNORMAL_FF_CD)

#Energy:
XOM = DATA[DATA.PERMNO == 11850]
EPD = DATA[DATA.PERMNO == 86223]
TRP = DATA[DATA.PERMNO == 67774]
EF = [XOM, EPD, TRP]
Energy = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'], how = 'outer'), EF).dropna()
Energy["PR"] = 0.33*Energy["Returns_x"] + 0.33*Energy["Returns_y"] + 0.33*Energy["Returns"]

#Change the index of the dataframes to Dates and drop some columns
Energy.index = Merge.index
Energy = Energy.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_E = Merge["ER"] = Energy["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_E = Merge[["MRP","SMB","HML"]]
Z_E = sm.add_constant(X_E)
Z_E.columns = ["alpha","beta1","beta2","beta3"]
MODEL_E = sm.OLS(Y_E,Z_E)
RESULTS_E = MODEL_E.fit()
print(RESULTS_E.summary())

#Acquire betas and alpha from the regression
ALPHA_E = RESULTS_E.params[0]
BETA1_E = RESULTS_E.params[1]
BETA2_E = RESULTS_E.params[2]
BETA3_E = RESULTS_E.params[3]

#Find the estimated FF returns
FF_RETURNS_E = ALPHA_E + BETA1_E*Merge["MRP"] + BETA2_E*Merge["SMB"] + BETA3_E*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_E = Merge.loc[Merge.index < "31-12-2019"]
Y_RED_E = REDUCED_E["ER"]
X_RED_E = REDUCED_E[["MRP","SMB","HML"]]
Z_RED_E = sm.add_constant(X_RED_E)
Z_RED_E.columns = ["alpha","beta1","beta2","beta3"]
MODEL_RED_E = sm.OLS(Y_RED_E,Z_RED_E)
RESULTS_RED_E = MODEL_RED_E.fit()
print(RESULTS_RED_E.summary())

#Acquire the pre-covid alpha and betas
ALPHA_RED_E = RESULTS_RED_E.params.alpha
BETA1_RED_E = RESULTS_RED_E.params.beta1
BETA2_RED_E = RESULTS_RED_E.params.beta2
BETA3_RED_E = RESULTS_RED_E.params.beta3

#Create covid database 
COVID_E = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_RETURNS_COVID_E = ALPHA_RED_E + BETA1_RED_E*(COVID_E["MRP"]) + BETA2_RED_E*(COVID_E["SMB"]) + BETA3_RED_E*(COVID_E["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_RETURNS_RED_E = FF_RETURNS_E[FF_RETURNS_E.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_E = FF_RETURNS_RED_E - FF_RETURNS_COVID_E
Abnormal_FF_sum_E = sum(ABNORMAL_FF_E)

#Financial Services:
JPM = DATA[DATA.PERMNO == 47896]
BAC = DATA[DATA.PERMNO == 59408]
WFC = DATA[DATA.PERMNO == 38703]
FSF = [JPM, BAC, WFC]
FinancialServices = reduce(lambda  left,right: pd.merge(left,right,on=['Dates'], how = 'outer'), FSF)
FinancialServices["PR"] = 0.33*FinancialServices["Returns_x"] + 0.33*FinancialServices["Returns_y"] + 0.33*FinancialServices["Returns"]

#Change the index of the dataframes to Dates and drop some columns
FinancialServices.index = Merge.index
FinancialServices = FinancialServices.drop(columns=["Dates","PERMNO","PERMNO_x","PERMNO_y"])

#Define y and x for the regression:
Y_FS = Merge["ER"] = FinancialServices["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_FS = Merge[["MRP","SMB","HML"]]
Z_FS = sm.add_constant(X_FS)
Z_FS.columns = ["alpha","beta1","beta2","beta3"]
MODEL_FS = sm.OLS(Y_FS,Z_FS)
RESULTS_FS = MODEL_FS.fit()
print(RESULTS_FS.summary())

#Acquire betas and alpha from the regression
ALPHA_FS = RESULTS_FS.params[0]
BETA1_FS = RESULTS_FS.params[1]
BETA2_FS = RESULTS_FS.params[2]
BETA3_FS = RESULTS_FS.params[3]

#Find the estimated FF returns
FF_RETURNS_FS = ALPHA_FS + BETA1_FS*Merge["MRP"] + BETA2_FS*Merge["SMB"] + BETA3_FS*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_FS = Merge.loc[Merge.index < "31-12-2019"]
Y_RED_FS = REDUCED_FS["ER"]
X_RED_FS = REDUCED_FS[["MRP","SMB","HML"]]
Z_RED_FS = sm.add_constant(X_RED_FS)
Z_RED_FS.columns = ["alpha","beta1","beta2","beta3"]
MODEL_RED_FS = sm.OLS(Y_RED_FS,Z_RED_FS)
RESULTS_RED_FS = MODEL_RED_FS.fit()
print(RESULTS_RED_FS.summary())

#Acquire the pre-covid alpha and betas
ALPHA_RED_FS = RESULTS_RED_FS.params.alpha
BETA1_RED_FS = RESULTS_RED_FS.params.beta1
BETA2_RED_FS = RESULTS_RED_FS.params.beta2
BETA3_RED_FS = RESULTS_RED_FS.params.beta3

#Create covid database 
COVID_FS = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_RETURNS_COVID_FS = ALPHA_RED_FS + BETA1_RED_FS*(COVID_FS["MRP"]) + BETA2_RED_FS*(COVID_FS["SMB"]) + BETA3_RED_FS*(COVID_FS["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_RETURNS_RED_FS = FF_RETURNS_FS[FF_RETURNS_FS.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_FS = FF_RETURNS_RED_FS - FF_RETURNS_COVID_FS
Abnormal_FF_sum_FS = sum(ABNORMAL_FF_FS)

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
Y_H = Merge["ER"] = Healthcare["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_H = Merge[["MRP","SMB","HML"]]
Z_H = sm.add_constant(X_H)
Z_H.columns = ["alpha","beta1","beta2","beta3"]
MODEL_H = sm.OLS(Y_H,Z_H)
RESULTS_H = MODEL_H.fit()
print(RESULTS_H.summary())

#Acquire betas and alpha from the regression
ALPHA_H = RESULTS_H.params[0]
BETA1_H = RESULTS_H.params[1]
BETA2_H = RESULTS_H.params[2]
BETA3_H = RESULTS_H.params[3]

#Find the estimated FF returns
FF_RETURNS_H = ALPHA_H + BETA1_H*Merge["MRP"] + BETA2_H*Merge["SMB"] + BETA3_H*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_H = Merge.loc[Merge.index < "31-12-2019"]
Y_RED_H = REDUCED_H["ER"]
X_RED_H = REDUCED_H[["MRP","SMB","HML"]]
Z_RED_H = sm.add_constant(X_RED_H)
Z_RED_H.columns = ["alpha","beta1","beta2","beta3"]
MODEL_RED_H = sm.OLS(Y_RED_H,Z_RED_H)
RESULTS_RED_H = MODEL_RED_H.fit()
print(RESULTS_RED_H.summary())

#Acquire the pre-covid alpha and betas
ALPHA_RED_H = RESULTS_RED_H.params.alpha
BETA1_RED_H = RESULTS_RED_H.params.beta1
BETA2_RED_H = RESULTS_RED_H.params.beta2
BETA3_RED_H = RESULTS_RED_H.params.beta3

#Create covid database 
COVID_H = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_RETURNS_COVID_H = ALPHA_RED_H + BETA1_RED_H*(COVID_H["MRP"]) + BETA2_RED_H*(COVID_H["SMB"]) + BETA3_RED_H*(COVID_H["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_RETURNS_RED_H = FF_RETURNS_H[FF_RETURNS_H.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_H = FF_RETURNS_RED_H - FF_RETURNS_COVID_H
Abnormal_FF_sum_H = sum(ABNORMAL_FF_H)

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
Y_I = Merge["ER"] = Industrials["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_I = Merge[["MRP","SMB","HML"]]
Z_I = sm.add_constant(X_I)
Z_I.columns = ["alpha","beta1","beta2","beta3"]
MODEL_I = sm.OLS(Y_I,Z_I)
RESULTS_I = MODEL_I.fit()
print(RESULTS_I.summary())

#Acquire betas and alpha from the regression
ALPHA_I = RESULTS_I.params[0]
BETA1_I = RESULTS_I.params[1]
BETA2_I = RESULTS_I.params[2]
BETA3_I = RESULTS_I.params[3]

#Find the estimated FF returns
FF_RETURNS_I = ALPHA_I + BETA1_I*Merge["MRP"] + BETA2_I*Merge["SMB"] + BETA3_I*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_I = Merge.loc[Merge.index < "31-12-2019"]
Y_RED_I = REDUCED_I["ER"]
X_RED_I = REDUCED_I[["MRP","SMB","HML"]]
Z_RED_I = sm.add_constant(X_RED_I)
Z_RED_I.columns = ["alpha","beta1","beta2","beta3"]
MODEL_RED_I = sm.OLS(Y_RED_I,Z_RED_I)
RESULTS_RED_I = MODEL_RED_I.fit()
print(RESULTS_RED_I.summary())

#Acquire the pre-covid alpha and betas
ALPHA_RED_I = RESULTS_RED_I.params.alpha
BETA1_RED_I = RESULTS_RED_I.params.beta1
BETA2_RED_I = RESULTS_RED_I.params.beta2
BETA3_RED_I = RESULTS_RED_I.params.beta3

#Create covid database 
COVID_I = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_RETURNS_COVID_I = ALPHA_RED_I + BETA1_RED_I*(COVID_I["MRP"]) + BETA2_RED_I*(COVID_I["SMB"]) + BETA3_RED_I*(COVID_I["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_RETURNS_RED_I = FF_RETURNS_I[FF_RETURNS_I.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_I = FF_RETURNS_RED_I - FF_RETURNS_COVID_I
Abnormal_FF_sum_I = sum(ABNORMAL_FF_I)

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
Y_RE = Merge["ER"] = RealEstate["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_RE = Merge[["MRP","SMB","HML"]]
Z_RE = sm.add_constant(X_RE)
Z_RE.columns = ["alpha","beta1","beta2","beta3"]
MODEL_RE = sm.OLS(Y_RE,Z_RE)
RESULTS_RE = MODEL_RE.fit()
print(RESULTS_RE.summary())

#Acquire betas and alpha from the regression
ALPHA_RE = RESULTS_RE.params[0]
BETA1_RE = RESULTS_RE.params[1]
BETA2_RE = RESULTS_RE.params[2]
BETA3_RE = RESULTS_RE.params[3]

#Find the estimated FF returns
FF_RETURNS_RE = ALPHA_RE + BETA1_RE*Merge["MRP"] + BETA2_RE*Merge["SMB"] + BETA3_RE*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_RE = Merge.loc[Merge.index < "31-12-2019"]
Y_RED_RE = REDUCED_RE["ER"]
X_RED_RE = REDUCED_RE[["MRP","SMB","HML"]]
Z_RED_RE = sm.add_constant(X_RED_RE)
Z_RED_RE.columns = ["alpha","beta1","beta2","beta3"]
MODEL_RED_RE = sm.OLS(Y_RED_RE,Z_RED_RE)
RESULTS_RED_RE = MODEL_RED_RE.fit()
print(RESULTS_RED_RE.summary())

#Acquire the pre-covid alpha and betas
ALPHA_RED_RE = RESULTS_RED_RE.params.alpha
BETA1_RED_RE = RESULTS_RED_RE.params.beta1
BETA2_RED_RE = RESULTS_RED_RE.params.beta2
BETA3_RED_RE = RESULTS_RED_RE.params.beta3

#Create covid database 
COVID_RE = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_RETURNS_COVID_RE = ALPHA_RED_RE + BETA1_RED_RE*(COVID_RE["MRP"]) + BETA2_RED_RE*(COVID_RE["SMB"]) + BETA3_RED_RE*(COVID_RE["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_RETURNS_RED_RE = FF_RETURNS_RE[FF_RETURNS_RE.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_RE = FF_RETURNS_RED_RE - FF_RETURNS_COVID_RE
Abnormal_FF_sum_RE = sum(ABNORMAL_FF_RE)

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
Y_T = Merge["ER"] = Technology["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_T = Merge[["MRP","SMB","HML"]]
Z_T = sm.add_constant(X_T)
Z_T.columns = ["alpha","beta1","beta2","beta3"]
MODEL_T = sm.OLS(Y_T,Z_T)
RESULTS_T = MODEL_T.fit()
print(RESULTS_T.summary())

#Acquire betas and alpha from the regression
ALPHA_T = RESULTS_T.params[0]
BETA1_T = RESULTS_T.params[1]
BETA2_T = RESULTS_T.params[2]
BETA3_T = RESULTS_T.params[3]

#Find the estimated FF returns
FF_TTURNS_T = ALPHA_T + BETA1_T*Merge["MRP"] + BETA2_T*Merge["SMB"] + BETA3_T*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_T = Merge.loc[Merge.index < "31-12-2019"]
Y_TD_T = REDUCED_T["ER"]
X_TD_T = REDUCED_T[["MRP","SMB","HML"]]
Z_TD_T = sm.add_constant(X_TD_T)
Z_TD_T.columns = ["alpha","beta1","beta2","beta3"]
MODEL_TD_T = sm.OLS(Y_TD_T,Z_TD_T)
RESULTS_TD_T = MODEL_TD_T.fit()
print(RESULTS_TD_T.summary())

#Acquire the pre-covid alpha and betas
ALPHA_TD_T = RESULTS_TD_T.params.alpha
BETA1_TD_T = RESULTS_TD_T.params.beta1
BETA2_TD_T = RESULTS_TD_T.params.beta2
BETA3_TD_T = RESULTS_TD_T.params.beta3

#Create covid database 
COVID_T = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_TTURNS_COVID_T = ALPHA_TD_T + BETA1_TD_T*(COVID_T["MRP"]) + BETA2_TD_T*(COVID_T["SMB"]) + BETA3_TD_T*(COVID_T["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_TTURNS_TD_T = FF_TTURNS_T[FF_TTURNS_T.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_T = FF_TTURNS_TD_T - FF_TTURNS_COVID_T
Abnormal_FF_sum_T = sum(ABNORMAL_FF_T)

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
Y_U = Merge["ER"] = Utilities["PR"] - Merge["RF"]
Merge["MRP"] =  Merge["S&P500"] -  Merge["RF"]
X_U = Merge[["MRP","SMB","HML"]]
Z_U = sm.add_constant(X_U)
Z_U.columns = ["alpha","beta1","beta2","beta3"]
MODEL_U = sm.OLS(Y_U,Z_U)
RESULTS_U = MODEL_U.fit()
print(RESULTS_U.summary())

#Acquire betas and alpha from the regression
ALPHA_U = RESULTS_U.params[0]
BETA1_U = RESULTS_U.params[1]
BETA2_U = RESULTS_U.params[2]
BETA3_U = RESULTS_U.params[3]

#Find the estimated FF returns
FF_UTURNS_U = ALPHA_U + BETA1_U*Merge["MRP"] + BETA2_U*Merge["SMB"] + BETA3_U*Merge["HML"]

#EVENT STUDY Code
#Create pre-covid database and run OLS regression as before with it
REDUCED_U = Merge.loc[Merge.index < "31-12-2019"]
Y_UD_U = REDUCED_U["ER"]
X_UD_U = REDUCED_U[["MRP","SMB","HML"]]
Z_UD_U = sm.add_constant(X_UD_U)
Z_UD_U.columns = ["alpha","beta1","beta2","beta3"]
MODEL_UD_U = sm.OLS(Y_UD_U,Z_UD_U)
RESULTS_UD_U = MODEL_UD_U.fit()
print(RESULTS_UD_U.summary())

#Acquire the pre-covid alpha and betas
ALPHA_UD_U = RESULTS_UD_U.params.alpha
BETA1_UD_U = RESULTS_UD_U.params.beta1
BETA2_UD_U = RESULTS_UD_U.params.beta2
BETA3_UD_U = RESULTS_UD_U.params.beta3

#Create covid database 
COVID_U = Merge.loc[Merge.index > "31-12-2019"]

#Estimate the FF returns for the Covid database
FF_UTURNS_COVID_U = ALPHA_UD_U + BETA1_UD_U*(COVID_U["MRP"]) + BETA2_UD_U*(COVID_U["SMB"]) + BETA3_UD_U*(COVID_U["HML"])

#Acquire the Actual FF returns for covid era that were estimated previously
FF_UTURNS_UD_U = FF_UTURNS_U[FF_UTURNS_U.index > "31-12-2019"]

#Estimate Abnormal returns
ABNORMAL_FF_U = FF_UTURNS_UD_U - FF_UTURNS_COVID_U
Abnormal_FF_sum_U = sum(ABNORMAL_FF_U)

#Create Dataframe containing all the abnormal returns
Dict = [ABNORMAL_FF_BM,ABNORMAL_FF_CS,ABNORMAL_FF_CC,ABNORMAL_FF_CD,ABNORMAL_FF_E,ABNORMAL_FF_FS,ABNORMAL_FF_H,ABNORMAL_FF_I,ABNORMAL_FF_RE,ABNORMAL_FF_T,ABNORMAL_FF_U]
Di = pd.concat(Dict, axis=1)
x ={'Abnormal Returns' : [Abnormal_FF_sum_BM,Abnormal_FF_sum_CS,Abnormal_FF_sum_CC,Abnormal_FF_sum_CD,Abnormal_FF_sum_E,Abnormal_FF_sum_FS,Abnormal_FF_sum_H,Abnormal_FF_sum_I,Abnormal_FF_sum_RE,Abnormal_FF_sum_T,Abnormal_FF_sum_U], 'Sector' : ['Basic Materials','Communication Services','Consumer Cyclical','Consumer Defensive','Energy','Financial Services','Healthcare','Industrials','Real Estate','Technology','Utilities']}
Di.columns = columns = x['Sector']

#Plotting of Abnormal Returns
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')
figure(figsize=(14, 10), dpi=100)
BM, = plt.plot(ABNORMAL_FF_BM)
CS, = plt.plot(ABNORMAL_FF_CS)
CC, = plt.plot(ABNORMAL_FF_CC)
CD, = plt.plot(ABNORMAL_FF_CD)
EN, = plt.plot(ABNORMAL_FF_E)
FS, = plt.plot(ABNORMAL_FF_FS)
HE, = plt.plot(ABNORMAL_FF_H)
IN, = plt.plot(ABNORMAL_FF_I)
RE, = plt.plot(ABNORMAL_FF_RE)
TE, = plt.plot(ABNORMAL_FF_T)
UT, = plt.plot(ABNORMAL_FF_U)
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
x1 = pd.DataFrame(x)
x1 = x1.sort_values(by=['Abnormal Returns'])
x1.reset_index(inplace=True)
x1 = x1.drop(columns = "index")
x1['colors'] = ['red' if x < 0 else 'green' for x in x1['Abnormal Returns']]
plt.figure(figsize=(14,10), dpi= 100)
plt.hlines(y = x1.index, xmin=0, xmax = x1['Abnormal Returns'], color=x1.colors, alpha=0.5, linewidth=5)
plt.ylabel("Sector", fontproperties=font, fontsize= 22)
plt.xlabel("Abnormal Returns", fontproperties=font, fontsize= 22)
plt.yticks(x1.index, x1.Sector, fontsize=12)
plt.title('Cumulated Abnormal Returns', fontproperties=font, fontsize= 26)
plt.grid(linestyle='--', alpha=0.5)
plt.show()

Mean = x1.mean()
Var = x1.var()


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
Corr_Pearson = [corr1_BM,corr1_CS,corr1_CC,corr1_CD,corr1_E,corr1_FS,corr1_H,corr1_I,corr1_RE,corr1_T,corr1_U]
Corr_Spearman = [corr2_BM,corr2_CS,corr2_CC,corr2_CD,corr2_E,corr2_FS,corr2_H,corr2_I,corr2_RE,corr2_T,corr2_U]
X_axis = np.arange(len(x['Sector']))
fig, ax = plt.subplots(figsize=(30, 10), dpi=100)
plt.bar(X_axis - 0.2, Corr_Pearson, 0.4, label = 'Pearson')
plt.bar(X_axis + 0.2, Corr_Spearman, 0.4, label = 'Spearman')
plt.xticks(X_axis, x['Sector'])
plt.xlabel("Sectors")
plt.ylabel("Correlations")
plt.title("Autocorrelation with Oxcford Government Response Tracker")
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.show()