#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MG-GY 8401 (MOT) Project

Group 6
Bhosale, Parnika
Ruan, Tinghao
Zhou, Yichun
Teng, Huiyi
Zhang, Yanbin
Zhou, Zixin

"""
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from statsmodels.tsa.stattools import adfuller
pd.core.common.is_list_like = pd.api.types.is_list_like
import matplotlib.pyplot as plt
from datetime import datetime
import time
import seaborn as sns
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from collections import Iterable
from pandas import Series
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from hurst import compute_Hc, random_walk
#read data
#eurusd = pd.read_csv('EUR_CNY Historical Data.csv', header=0, index_col="Date")
#Read data from 10 different time series of currency pairs

eurgbp= pd.read_csv('EUR_GBP Historical Data.csv', header=0, index_col="Date")
eurcny= pd.read_csv('EUR_CNY Historical Data.csv', header=0, index_col="Date")
eurhkd= pd.read_csv('EUR_HKD Historical Data.csv', header=0, index_col="Date")
eurjpy= pd.read_csv('EUR_JPY Historical Data.csv', header=0, index_col="Date")
eurusd= pd.read_csv('EUR_USD Historical Data.csv', header=0, index_col="Date")
usdcad= pd.read_csv('USD_CAD Historical Data.csv', header=0, index_col="Date")
usdcny= pd.read_csv('USD_CNY Historical Data.csv', header=0, index_col="Date")
usdgbp= pd.read_csv('USD_GBP Historical Data.csv', header=0, index_col="Date")
usdhkd= pd.read_csv('USD_HKD Historical Data.csv', header=0, index_col="Date")
usdjpy= pd.read_csv('USD_JPY Historical Data.csv', header=0, index_col="Date")

#eurusd_price_serie = eurusd['Price']
#extract price
eurgbp_price_serie= eurgbp['Price']
eurcny_price_serie=eurcny['Price']
eurhkd_price_serie=eurhkd['Price']
eurjpy_price_serie=eurjpy['Price']
eurusd_price_serie=eurusd['Price']
usdcad_price_serie=usdcad['Price']
usdcny_price_serie=usdcny['Price']
usdgbp_price_serie=usdgbp['Price']
usdhkd_price_serie=usdhkd['Price']
usdjpy_price_serie=usdjpy['Price']

# print(eurusd)
# eurusd_price_clean_list = eurusd['Price']
# eurusd_price_clean_list = eurusd_price_clean_list[::-1]
# print(eurusd_price_clean_list)
#eurusd.columns = ['price', 'open', 'high', 'low', 'percentage_change']
eurgbp.columns = ['price', 'open', 'high', 'low', 'percentage_change']
eurgbp_price = eurgbp[['price']]
eurcny.columns = ['price', 'open', 'high', 'low', 'percentage_change']
eurcny_price = eurcny[['price']]
eurhkd.columns = ['price', 'open', 'high', 'low', 'percentage_change']
eurhkd_price = eurhkd[['price']]
eurjpy.columns = ['price', 'open', 'high', 'low', 'percentage_change']
eurjpy_price = eurjpy[['price']]
eurusd.columns = ['price', 'open', 'high', 'low', 'percentage_change']
eurusd_price = eurusd[['price']]
usdcad.columns = ['price', 'open', 'high', 'low', 'percentage_change']
usdcad_price = usdcad[['price']]
usdcny.columns = ['price', 'open', 'high', 'low', 'percentage_change']
usdcny_price = usdcny[['price']]
usdgbp.columns = ['price', 'open', 'high', 'low', 'percentage_change']
usdgbp_price = usdgbp[['price']]
usdhkd.columns = ['price', 'open', 'high', 'low', 'percentage_change']
usdhkd_price = usdhkd[['price']]
usdjpy.columns = ['price', 'open', 'high', 'low', 'percentage_change']
usdjpy_price = usdjpy[['price']]
eurusd_price = eurusd[['price']]
eurusd_price = eurusd_price[::-1]
#print(eurusd_price)

#############################     volatility    #############################
print("-----------------Volatility----------------")
print("\n")
print()
# Initiate the V list, to store all the Volatility from all the data files
v = []
#Function to get the volatility of a list
def volatility(price): 
    #iterate in the list, set i as a index
    for i in range(len(price)):
        # check whether there is enough data (100 float numbers) to process 
        # the later calculation
        if len(price)-i >= 100:
            # using the NumPy std function to find the standard deviation of 
            # using the Numpy mean function to find the average 
            # the final step of the calculation would be using standard deviation
            # divide the average to get the volatility
            # store into the list v
            v.append(np.std(price[i:i+100])/np.mean(price[i:i+100]))  
    # return the list v
    return v

# call the volatility function for each currency pair 
eurgbp_volatility = volatility(eurgbp_price_serie)
eurcny_volatility= volatility(eurcny_price_serie)
eurhkd_volatility= volatility(eurhkd_price_serie)
eurjpy_volatility= volatility(eurjpy_price_serie)
eurusd_volatility= volatility(eurusd_price_serie)
usdcad_volatility= volatility(usdcad_price_serie)
usdcny_volatility= volatility(usdcny_price_serie)
usdgbp_volatility= volatility(usdgbp_price_serie)
usdhkd_volatility= volatility(usdhkd_price_serie)
usdjpy_volatility= volatility(usdjpy_price_serie)

#eurusd_volatilitty_test2 = volatility(eurusd_price_serie)
#print(usdjpy_volatility)

##############################################################################


#############################   Fractal Dimension    #########################
print("-----------------Factal Dimension----------------")
print("\n")
print()

# def FD(price_data):
#     fd_list = []
#     #fd_list = []
#     # hurst_list = []
#     for i in range(len(price_data)):
#         if len(price_data)-i >=100:
#             eurusd_price_2 = price_data[i:i+100]
            
#             lag1, lag2 = 2, 20
#             lags = range(lag1, lag2)      
#             tau = [sqrt(std(subtract(eurusd_price_2[lag:], eurusd_price_2[:-
#             lag]))) for lag in lags]
#             m = polyfit(log(lags), log(tau), 1)
#             if m[0] < 0 and m[1]<0:
#                 m[0] = 0
#             elif m[0] < 0  and m[1]>0:
#                 m[0] = m[1]
#             hurst = m[0] * 2
#                     # hurst_list.append(hurst[0])
#             # print('\nhurst = ', hurst[0])           
#             # standard_dev = eurusd_price_2['price'].std()
#             # print('\n\nStandard deviation = ', standard_dev)            
#             fractal_d = 2 - hurst[0]            
#             # print('\n\nFractal dimension = ', fractal_d)
#             fd_list.append(fractal_d)                    
#     return fd_list
# Code above is one way to compute Fractal Dimension provided by the PDF file of
# non-graded homework

# However a better way we found is to use the hurst package and set kind='price

# Initiate the FD list, to store all the Fractal Dimension from all the data files
fd_list = []
#Function to get the Fractal Dimension of a list
def FD(price_data):
    #iterate in the list, set i as a index
    for i in range(len(price_data)):
        # check whether there is enough data (100 float numbers) to process 
        # the later calculation
        if len(price_data)-i >=100:
            # store the list with 100 element
            eurusd_price_2 = price_data[i:i+100]
            # use compute_Hc function to get the hurst
            # set kind='price' and simplified=True
            hurst = compute_Hc(eurusd_price_2,kind='price',simplified=True)
            # caculate Fractal Dimension use 2 minus the first element in hurst
            FD = 2 - hurst[0]
            # store the Fractal Dimension into a list
            fd_list.append(FD)
    # return the list of Fractal Dimension
    return fd_list

# call the Fractal Dimension function for each currency pair 
eurgbp_fd = FD(eurgbp_price)
eurcny_fd = FD(eurcny_price)
eurhkd_fd = FD(eurhkd_price)
eurjpy_fd = FD(eurjpy_price)
eurusd_fd = FD(eurusd_price)
usdcad_fd = FD(usdcad_price)
usdcny_fd = FD(usdcny_price) 
usdgbp_fd = FD(usdgbp_price)
usdhkd_fd = FD(usdhkd_price)
usdjpy_fd = FD(usdjpy_price)

#print(fd_list)
# assert len(v) == len(fd_list)

# if you want see the scatter plot of Fractal Dimension vs volatility,
# uncomment the code below
# plt.scatter(fd_list,v)
##############################################################################
##########################   Conservation Law   ##############################
##############################################################################

print("\n")
print("-----------------Linear Regression ----------------")
print("\n")
# initiate a new list for later store use
two_d = []
#helper function to get Fractal Dimension to a formatted list
def two_d_list(sample_list):
    #iterate in the list of Fractal Dimension
    for i in sample_list:
        #append into the created list
        two_d.append([i])
    # return the list
    return two_d

#helper function to get the predicted value of a LinearRegression model
#need to pass in the list of x, the coefficient and the intercept
def predict_list(x_list,a,b):
    #new list for later store use
    p_list = []
    #iterate in the list of x
    for i in x_list:
        # predicted y would be coefficient * the current x add intercept
        y = a * i + b
        # store into the list
        p_list.append(y)
    #return the list
    return p_list


# use two_d_list and numpy array to help make the correct format of the list
# set x as the Fractal Dimension
x = np.array(two_d_list(fd_list))
# set y as Volatility
y = v

##########################   Linear Regression ##############################
# use sklearn.linear_model LinearRegression to model the Fractal Dimension and
# Volatility
model = LinearRegression()
# fit the model of Fractal Dimension and Volatility
model.fit(x,y)
# print out the coefficient and intercept of LinearRegression model
print("Linear Coef: ",model.coef_[0])
print("Linear Intercept: ",model.intercept_)
# Linear Coef:  -0.024525304962776343
# Linear Intercept:  0.04803921559960725

#use predict_list function to get the LinearRegression prediction of volatility
V_predict = predict_list(fd_list,model.coef_[0], model.intercept_)

# find the R_square of the prediction compare with actual volatility
print("Linear Regression R_square: ",r2_score(v, V_predict))
# R-square: 0.2283191674677999

# find the mean square error of the prediction compare with actual volatility
mse = np.square(np.subtract(v,V_predict)).mean()
# print out the mse
print("Linear Regression MSE: ",mse)
# Linear Regression MSE:  3.13772186574459e-05
print("\n")

# to plot the scatter of FD and V
# also plot the LinearRegression line
x_test = np.linspace(1,1.8)
y_prep = model.predict(x_test[:,None])
plt.scatter(x,y)
plt.plot(x_test,y_prep,'r')
plt.legend(['Linear Predicted line','Observed data'])
plt.savefig('Linear.png')
plt.show()

# caculate for the conservation law based on LinearRegression
#aV + bFD = c
#Assume c = 1
b = 1/model.intercept_
a = -1 * b * model.coef_
#print out the conservation equation of Fractal Dimension and Volatility
print("Conservation Law Generated by Linear Regression Model")
print(a[0],"FD + ",b,"V = 1")
# 0.5105267572890356 FD +  20.816326568166854 V = 1

##########################    Ridge Regression  ##############################
print("\n")
print("-----------------Ridge Regression ----------------")
print("\n")

# use sklearn.linear_model Ridge to model the Fractal Dimension and Volatility
RR=Ridge()
# fit the Ridge model of Fractal Dimension and Volatility
Rireg = RR.fit(x,y)
# store the coefficient of the fitted model
w1 = Rireg.coef_
# store the intercept of the fitted model
w0 = Rireg.intercept_
print("Ridge Coef: ",w1[0])
print("Ridge Intercept: ",w0)
# Ridge Coef:  -0.02427044459967178
# Ridge Intercept:  0.047665767922544865

#use predict_list function to get the Ridge prediction of volatility
V_RR_predict = predict_list(fd_list,w1[0], w0)

# find the R_square of the prediction compare with actual volatility
print("Ridge Regression R_square: ",r2_score(v, V_RR_predict))
#R-Square: 0.2282945117225309

# find the mean square error of the prediction compare with actual volatility
Ridge_MSE = np.square(np.subtract(v,V_RR_predict)).mean()
# print out the mse of using Ridge model
print("Ridge Regression MSE: ",Ridge_MSE)
print("\n")
# Ridge Regression MSE:  3.137822118164742e-05

# to plot the scatter of FD and V
# also plot the LinearRegression line
x_test = np.linspace(1,1.8)
# x_test = np.linspace(1.05,1.2)
y_prep = RR.predict(x_test[:,None])
plt.scatter(x,y)
plt.plot(x_test,y_prep,'r')
plt.legend(['Ridge Predicted line','Observed data'])
plt.savefig('Ridge.png')
plt.show()


# caculate for the conservation law based on LinearRegression
#aV + bFD = c
#Assume c = 1
b1 = 1/RR.intercept_
a1 = -1 * b * RR.coef_
#print out the conservation equation of Fractal Dimension and Volatility
print("Conservation Law Generated by Ridge Regression Model")
print(a1[0],"FD + ",b1,"V = 1")
# 0.5052215007413694 FD +  20.979416541132068 V = 1

# ##########################   Lasso Regression  ##############################
print("\n")
print("-----------------Lasso Regression ----------------")
print("\n")

# use sklearn.linear_model Lasso to model the Fractal Dimension and Volatility
La=Lasso()
# fit the Lasso model of Fractal Dimension and Volatility
Lareg=La.fit(x,y)
# store the coefficient of the fitted model
Lw1 = La.coef_
# store the intercept of the fitted model
Lw0 = La.intercept_

print("Lasso Coef: ",Lw1[0])
print("Lasso Intercept: ",Lw0)
# Lasso Coef:  -0.0
# Lasso Intercept:  0.012102210503849494

#use predict_list function to get the Lasso prediction of volatility
V_LS_predict = predict_list(fd_list,Lw1[0],Lw0)
print("Lasso Regression R_square: ",r2_score(v, V_LS_predict))
#R Square: 0.0

# find the mean square error of the prediction compare with actual volatility
lasso_MSE = np.square(np.subtract(v,V_LS_predict)).mean()
# print mse
print("Lasso Regression MSE: ",lasso_MSE)
# Lasso Regression MSE:  4.0660876018501615e-05

# The predict line using lasso does not make any sense, if you want to see it, 
#Please uncomment the codes below
# x_test = np.linspace(1,1.8)
# # x_test = np.linspace(1.05,1.2)
# y_prep = La.predict(x_test[:,None])
# plt.scatter(x,y)
# plt.plot(x_test,y_prep,'r')
# plt.legend(['Lasso Predicted line','Observed data'])
# plt.show()

# Conservation Law Generated by Lasso Regression Model does not make any sense
# if you want to see it, Please uncomment the codes below
# b2 = 1/La.intercept_
# a2 = -1 * b * La.coef_
# #print(a,b)
# print("Conservation Law Generated by Lasso Regression Model")
# print(a2[0],"FD + ",b2,"V = 1")


##### Another way to find MSE
##### source: https://towardsdatascience.com/how-to-perform-lasso-and-ridge-regression-in-python-3b3b75541ad8
################################## Linear_Reg MSE##############################
# print("-----------------Linear_Regression_MSE----------------")
# print("\n")
# print()
# two_d = []
# def two_d_list(sample_list):
#     #two_d = []
#     for i in sample_list:
#         two_d.append([i])
#     return two_d
# # two_d_list([1,2,3])
# x = np.array(two_d_list(fd_list))
# y = v

# lin_reg = LinearRegression()
# MSE= cross_val_score(lin_reg,x,y, scoring='neg_mean_squared_error', cv=6170)
# mean_MSE=np.mean(MSE)
# print("Linear_Reg_mean_MSE:",mean_MSE)

# ################################## Ridge_Reg MSE###############################
# print("-----------------Ridge_Regression_MSE----------------")
# print("\n")
# print()
# ridge = Ridge()
# parameters ={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
# ridge_regressor=GridSearchCV(ridge, parameters, scoring = 'neg_mean_squared_error', cv=6170)
# ridge_regressor.fit(x,y)
# print("Best Parameters:",ridge_regressor.best_params_)
# print("Regressors(MSE):",ridge_regressor.best_score_)

# ##################################lasso_Reg MSE################################
# print("-----------------Lasso_Regression_MSE----------------")
# print("\n")
# print()
# lasso = Lasso()
# parameters ={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
# lasso_regressor=GridSearchCV(lasso, parameters, scoring = 'neg_mean_squared_error', cv=6170)
# lasso_regressor.fit(x,y)
# print("Best Parameters:",lasso_regressor.best_params_)
# print("Regressors(MSE):",lasso_regressor.best_score_)


