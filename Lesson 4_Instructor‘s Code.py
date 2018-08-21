# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 22:04:10 2018

@author: Xie Wenjun 
Copyright: Nanyang Technological University
"""
#%%
#Lesson 4

# Scipy的子包stats
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

#%%
# 基础统计数据
s = np.random.normal(loc=0,scale=1,size=1000)
print (len(s))
#描述性统计数据
print("Mean : {0:8.6f}".format(s.mean()))
print("Minimum : {0:8.6f}".format(s.min()))
print("Maximum : {0:8.6f}".format(s.max()))
print("Variance : {0:8.6f}".format(s.var()))
print("Std. deviation : {0:8.6f}".format(s.std()))

n, min_max, mean, var, skew, kurt = stats.describe(s)
print("Number of elements: {0:d}".format(n))
print("Minimum: {0:8.6f} Maximum: {1:8.6f}".format(min_max[0], min_max[1]))
print("Mean: {0:8.6f}".format(mean))
print("Variance: {0:8.6f}".format(var))
print("Skew : {0:8.6f}".format(skew))
print("Kurtosis: {0:8.6f}".format(kurt))

#%%
#概率分布 PDF和 PMF
stats.norm.pdf(0, loc=0.0, scale=1.0)
stats.norm.pdf([-0.1, 0.0, 0.1], loc=0.0, scale=1.0)
def binom_pmf(n, p):
    x = range(n+1)
    y = stats.binom.pmf(x, n, p)
    plt.plot(x,y,"o", color="black")
    plt.axis([-(max(x)-min(x))*0.05, max(x)*1.05, -0.01, max(y)*1.10])
    plt.xticks(x)
    plt.title("Binomial distribution PMF for tries = {0} & p ={1}".format(n,p))
    plt.xlabel("Variate")
    plt.ylabel("Probability")
    plt.show()
binom_pmf(n=20, p=0.1)

#%%
#概率分布 CDF
stats.norm.cdf(0.0, loc=0.0, scale=1.0)
def norm_cdf(mean, std):
    x = sp.linspace(-3*std, 3*std, 50)
    y = stats.norm.cdf(x, loc=mean, scale=std)
    plt.plot(x,y, color="black")
    plt.xlabel("Variate")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF for Gaussian of mean = {0} & std. deviation = {1}".format(
            mean, std))
    plt.show()
norm_cdf(0,1)

#%%
#概率分布 PPF (quantile)
stats.norm.ppf(0.5, loc=0.0, scale=1.0)
def norm_ppf(mean, std):
    x = sp.linspace(0, 1.0, 100)
    y = stats.norm.ppf(x, loc=0, scale=1)
    plt.plot(x,y, color="black")
    plt.xlabel("Cumulative Probability")
    plt.ylabel("Variate")
    plt.title("PPF for Gaussian of mean = {0} & std. deviation = {1}".format(
               mean, std))
    plt.show()
norm_ppf(0,1)

#%%
# 分布拟合
from scipy.stats import norm
from numpy import linspace
from pylab import plot,show,hist,title
samp = norm.rvs(loc=0,scale=1,size=1000) 
param = norm.fit(samp)
x = linspace(-5,5,100)
pdf_fitted = norm.pdf(x,loc=param[0],scale=param[1])
pdf = norm.pdf(x)
title('Normal distribution')
plot(x,pdf_fitted,'r-',x,pdf,'b-')
hist(samp,normed=1,alpha=.3)
show()

#%%
# 正态检验
from scipy import stats
import numpy as np
pts = 1000
np.random.seed(1)
a = np.random.normal(0, 1, size=pts)
b = np.random.normal(2, 1, size=pts)
x = np.concatenate((a, b))
k2, p = stats.normaltest(x)
alpha = 1e-3
print("p = {:g}".format(p))
if p < alpha:
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
k2, p = stats.normaltest(b)
alpha = 1e-3
print("p = {:g}".format(p))
if p < alpha:
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")

#%%
# 样本均值t检验
from scipy import stats
np.random.seed(1)
rvs = stats.norm.rvs(size=50,loc=5,scale=10)
stats.ttest_1samp(rvs,5)
stats.ttest_1samp(rvs,0)

#%%
# 线性回归 - statsmodel.OLS
import statsmodels.api as sm
from sklearn import datasets
import numpy as np
import pandas as pd
data = datasets.load_boston() 
print (data.DESCR)
df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=["MEDV"])
X = df[["RM","CRIM"]]
y = target["MEDV"] 
X = sm.add_constant(X) 
model = sm.OLS(y, X).fit() 
predictions = model.predict(X)
model.summary()
dir(model)

#%%
# 线性回归 - sklearn.LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
boston = load_boston()
print (boston.keys())
print (boston.feature_names)
x = boston.data[:, np.newaxis, 5]
y = boston.target
lm = LinearRegression()   
lm.fit(x, y)  
print ('方程的确定性系数(R^2): %.2f' % lm.score(x, y))
plt.scatter(x, y, color='green')   
plt.plot(x, lm.predict(x), color='blue', linewidth=3)   
plt.xlabel('Average Number of Rooms per Dwelling (RM)')
plt.ylabel('Housing Price')
plt.title('2D Demo of Linear Regression')
plt.show()

#%%
# 线性回归 - NumPy.polyfit
import os
import pandas as pd
os.chdir('C:/Users/XIEW0/Desktop/Data Science with Python/第四课')
raw = pd.read_csv('source/tr_eikon_eod_data.csv',index_col=0, parse_dates=True)
spx = pd.DataFrame(raw['.SPX'])
np.round(spx.tail())
vix = pd.DataFrame(raw['.VIX'])
vix.info()
data = spx.join(vix)
data.tail()
data.plot(subplots=True, grid=True, style='b', figsize=(8, 6));
rets = np.log(data / data.shift(1)) 
rets.head()
rets.dropna(inplace=True)
rets.plot(subplots=True, grid=True, style='b', figsize=(8, 6));
import numpy as np
xdat = rets['.SPX'].values
ydat = rets['.VIX'].values
reg = np.polyfit(x=xdat, y=ydat, deg=1)
reg
plt.plot(xdat, ydat, 'r.')
ax = plt.axis()  # grab axis values
x = np.linspace(ax[0], ax[1] + 0.01)
plt.plot(x, np.polyval(reg, x), 'b', lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel('S&P 500 returns')
plt.ylabel('VIX returns')

#%%
# 时间序列模型： ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf
discfile = 'arima_data.xls'
data = pd.read_excel(discfile,index_col=0,parse_dates=True)
print(data.head())
print('\n Data Types:')
print(data.dtypes)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
data.plot()
plt.show()
plot_acf(data).show()
from statsmodels.tsa.stattools import adfuller as ADF
print(u'原始序列的ADF检验结果为：', ADF(data[u'销量']))

#%%
# 一阶差分
D_data = data.diff().dropna()
D_data.columns = [u'销量差分']
D_data.plot() #时序图
plt.show()
plot_acf(D_data).show()
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(D_data).show() 
print(u'差分序列的ADF检验结果为：', ADF(D_data[u'销量差分'])) 
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))

#%%
# 一阶差分
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)
diff1 = data.diff(1)
diff1.plot(ax=ax1)
# 二阶差分
fig = plt.figure(figsize=(12,8))
ax2= fig.add_subplot(111)
diff2 = data.diff(2)
diff2.plot(ax=ax2)
# 合适的p,q
dta = data.diff(1)[1:]
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig1 = sm.graphics.tsa.plot_acf(dta[u'销量'],lags=10,ax=ax1)
ax2 = fig.add_subplot(212)
fig2 = sm.graphics.tsa.plot_pacf(dta[u'销量'],lags=10,ax=ax2)

#%%
#模型
arma_mod20 = sm.tsa.ARMA(dta,(2,0)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
arma_mod01 = sm.tsa.ARMA(dta,(0,1)).fit()
print(arma_mod01.aic,arma_mod01.bic,arma_mod01.hqic)
arma_mod10 = sm.tsa.ARMA(dta,(1,0)).fit()
print(arma_mod10.aic,arma_mod10.bic,arma_mod10.hqic)
arma_mod11 = sm.tsa.ARMA(dta,(1,1)).fit()
print(arma_mod11.aic,arma_mod11.bic,arma_mod11.hqic)

arma_mod01.summary()
#%%
#残差QQ图
resid = arma_mod01.resid
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
#残差自相关检验
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_mod01.resid, lags=10, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_mod01.resid, lags=10, ax=ax2)
#D-W检验
print(sm.stats.durbin_watson(arma_mod01.resid.values))
# Ljung-Box检验
import numpy as np
r,q,p = sm.tsa.acf(resid.values, qstat=True)
datap = np.c_[range(1,36), r[1:], q, p]
table = pd.DataFrame(datap, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
#预测
predict_sunspots = arma_mod01.predict('2015-2-07', '2015-2-15', dynamic=True)
fig, ax = plt.subplots(figsize=(12, 8))
print(predict_sunspots)
predict_sunspots[0] += data['2015-02-06':][u'销量']
data=pd.DataFrame(data)
for i in range(len(predict_sunspots)-1):
    predict_sunspots[i+1]=predict_sunspots[i]+predict_sunspots[i+1]
print(predict_sunspots)
ax = data.ix['2015':].plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()

#%% GARCH(1,1) estimation
from arch import arch_model
garch11 = arch_model(dta, p=1, q=1)
res = garch11.fit(update_freq=10)
print(res.summary())

#%%
# 主成分分析（PCA）
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  
print(pca.components_)  

#%%
#股票指数
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()
from sklearn.decomposition import KernelPCA
symbols = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE',
           'BMW.DE', 'CBK.DE', 'CON.DE', 'DAI.DE', 'DB1.DE',
           'DBK.DE', 'DPW.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
           'FRE.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE', 'LHA.DE',
           'LIN.DE', 'LXS.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE',
           'SAP.DE', 'SDF.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE',
           '^GDAXI']
data = pd.DataFrame()
for sym in symbols:
    data[sym] = pdr.get_data_yahoo(sym, start='2016-1-1')['Adj Close']
data = data.dropna()
print('')
dax = pd.DataFrame(data.pop('^GDAXI'))
data[data.columns[:6]].head()
scale_function = lambda x: (x - x.mean()) / x.std()
pca = KernelPCA().fit(data.apply(scale_function))
len(pca.lambdas_)
pca.lambdas_[:10].round()
get_we = lambda x: x / x.sum()
get_we(pca.lambdas_)[:10]
get_we(pca.lambdas_)[:5].sum()

pca = KernelPCA(n_components=1).fit(data.apply(scale_function))
dax['PCA_1'] = pca.transform(data)
import matplotlib.pyplot as plt
dax.apply(scale_function).plot(figsize=(8, 4))

#%%
# 因子分析
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2)
iris_two_dim = fa.fit_transform(iris.data)
iris_two_dim[:5]
from matplotlib import pyplot as plt
f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
ax.scatter(iris_two_dim[:,0], iris_two_dim[:, 1], c=iris.target)
ax.set_title("Factor Analysis 2 Components")

#%%
