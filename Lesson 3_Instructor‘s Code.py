# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 22:04:10 2018

@author: Xie Wenjun 
Copyright: Nanyang Technological University
"""
#%%
#Lesson 3

#%%
# 如何使用pip进行安装python模块
!pip
!pip list
!pip install pandas
!pip install numpy
!pip install matplotlib

#%%
# 模块的导入
import numpy
import numpy as np
from numpy import array
from numpy import array as ay

#%%
# NumPy数组的创建与读取
arr1 = np.array([2,3,4])   
arr2 = np.array([(1.3,9,2.0),(7,6,1)])  
arr3 = np.zeros((2,3))    
arr4 = np.identity(3)   
arr5 = np.random.random(size = (2,3)) 
arr6 = np.arange(5,20,3)  
arr7 = np.linspace(0,2,9) 
arr7 = np.linspace(0,2,9,endpoint=False) 

#%%
# NumPy数组的参数
print (arr2.shape) 
print (arr2.ndim)  
print (arr2.size)  
print (arr2.dtype.name)  
print (type(arr2)) 
#%%
# 通过索引和切片访问数组元素
def f(x,y):
    return 10*x+y
arr8 = np.fromfunction(f,(4,3),dtype = int)
print (arr8)
print (arr8[1,2]) 
print (arr8[0:2,:])  
print (arr8[:,1])   
print (arr8[-1])     

# 通过迭代器访问数组元素
for row in arr8:
    print (row)

for element in arr8.flat:
    print (element)

type(arr8.flat)

#%%
# Numpy数组的运算
arr9 = np.array([[2,1],[1,2]])
arr10 = np.array([[1,2],[3,4]])
print (arr9 - arr10)  
print (arr9**2)
print (3*arr10)
print (arr9*arr10)  
print (np.dot(arr9,arr10))  
print (arr10.T)  
print (np.linalg.inv(arr10)) 
print (arr10.sum())  
print (arr10.sum(axis =0))  
print (arr10.sum(axis =1))  
print (arr10.max())  
print (arr10.max(axis =0))  
print (arr10.max(axis =1))
print (arr10.cumsum())  
print (arr10.cumsum(axis =0))  
print (arr10.cumsum(axis =1)) 
#另一种格式
print (np.sum(arr10,axis=0))

#%%
#通用函数
print (np.exp(arr9))     
print (np.sin(arr9))      
print (np.sqrt(arr9))     
print (np.add(arr9,arr10))  

#%%
# 合并
arr11 = np.vstack((arr9,arr10)) 
print (arr11)
arr12 = np.hstack((arr9,arr10)) 
print (arr12)
# 分割
list1 = np.hsplit(arr12,2)
print (list1)  
list2 = np.vsplit(arr11,2)
print (list2)  

#%%
#常用的功能
print(np.empty([3,4],dtype=int))
print(np.average(arr10))
print(np.var(arr9))
print(np.reshape(arr10,(1,4)))
print(np.eye(6))
print(np.transpose(arr10))
print(np.std(arr10))
print(np.cov(arr10))

#%%
#Pandas的导入
import numpy as np
import pandas as pd

#%%
# Pandas数据框（dataframe）
#使用字典创建
data = {'id': ['Jack', 'Sarah', 'Mike'],
        'age': [18, 35, 20],
        'cash': [10.53, 500.7, 13.6]}
df = pd.DataFrame(data)    
print (df)  
df2 = pd.DataFrame(data, 
      columns=['id', 'age', 'cash'],
      index=['one', 'two', 'three'])
print (df2)
print (df2['id'])

#%%
# 创建时间序列
a = np.random.standard_normal([9,4])
a.round(6)
df = pd.DataFrame(a)
df.columns = [['APPL','GOOG','FB','AMZN']]
print (df)
dates = pd.date_range('2015-1-1',periods =9, freq ='M')
print (dates)
df.index = dates
print(df)
dates = pd.date_range('2015-1-1',periods =200, freq ='B')
print(dates)
dates = pd.date_range('2015-1-1',periods =200, freq ='H')
print(dates)

#%% 
# 访问数据框
print(df.dtypes)
print(df.head())
print(df.tail())
print(df.index)
print(df.columns)
print(df.values)
print(df.describe())
df.sort_index(axis=0,ascending=False)
df.sort_index(axis=1,ascending=False)
df.sort_values('APPL',ascending=False)
pd.DataFrame.sort_values(df,'APPL')
df['APPL']
print(type(np.array(df['APPL'])))
df[0:3]
df['2015-01-31':'2015-05-31']
df['2015-01-31':'2015-05-15']
df['20150131':'20150515']
df['2015/01/31':'2015/05/15']
df.loc['2015/01/31':'2015/05/15',['APPL','AMZN']]
df.iloc[1,1]
df.iloc[1:3,1:3]
df[df.APPL>0]
del df['APPL']
print(df)
df['APPL'] = df['AMZN'] > 0
print(df)
del df['APPL']

#%%
# Pandas系列(series)
s = pd.Series({'a': 4, 'b': 9, 'c': 16}, name='number')
print (s)
print (s[0])
print (s[:3])
print (s['a'])
s['d'] = 25    
print (s)

df_new = df['GOOG']
type(df_new)

#%%
# 数据框的基本操作
df.sum()
df.mean()
df.cumsum()
df.describe()
np.sqrt(df)

#%%
# Pandas基本的文件读入读出
rows = 5000
a = np.random.standard_normal((rows,5))
a.round(4)
t = pd.date_range(start='2014/1/1',periods=rows,freq='H')
import os
print(os.getcwd())
os.chdir('C:/Users/XIEW0/Desktop/Data Science with Python/第三课')
print(os.getcwd())
path=os.getcwd()
csv_file=open(path+'\out_data.csv','w')
header = 'date,AAPL,GOOG,FB,AMZN,MSFT\n'
csv_file.write(header)
for t_variable, (AAPL_v,GOOG_v,FB_v,AMZN_v,MSFT_v) in zip(t,a):
    s = '%s,%f,%f,%f,%f,%f\n' % (t_variable,AAPL_v,GOOG_v,FB_v,AMZN_v,MSFT_v)
    csv_file.write(s)
csv_file.close()

csv_file = open(path+'\out_data.csv','r')
for i in range(5):
    print (csv_file.readline())
    
csv_file = open(path+'\out_data.csv','r')
content = csv_file.readlines()
csv_file.close()

#%%
# pandas自带功能
df=pd.DataFrame(a)
df.columns =['AAPL','GOOG','FB','AMZN','MSFT']
df.index = t
df.to_csv(path+'\direct_out_data.csv')
df_read = pd.read_csv('direct_input_data.csv',index_col='Date',parse_dates =True)

#%%
# Matplotlib图表绘制
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%
# 二维图
x = np.linspace(0, 10, 500)
dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off
fig, ax = plt.subplots()
line1, = ax.plot(x, np.sin(x), '--', linewidth=2,
                 label='Dashes set retroactively')
line1.set_dashes(dashes)
line2, = ax.plot(x, -1 * np.sin(x), dashes=[30, 5, 10, 5],
                 label='Dashes set proactively')
ax.legend(loc='lower right')
plt.show()

#%%
# 子图
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()

#%%
#文本文字信息
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

x = np.linspace(0.0, 5.0, 100)
y = np.cos(2*np.pi*x) * np.exp(-x)

plt.plot(x, y, 'k')
plt.title('Damped exponential decay', fontdict=font)
plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
plt.xlabel('time (s)', fontdict=font)
plt.ylabel('voltage (mV)', fontdict=font)

plt.subplots_adjust(left=0.15)
plt.show()

#%% 
# 坐标轴
np.random.seed(1)
# make up some data in the interval ]0, 1[
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))
# plot with various axes scales
fig, axs = plt.subplots(1, 2, sharex=True)
fig.subplots_adjust(left=0.08, right=0.98, wspace=0.3)
# linear
ax = axs[0]
ax.plot(x, y)
ax.set_yscale('linear')
ax.set_title('linear')
ax.grid(True)
# log
ax = axs[1]
ax.plot(x, y)
ax.set_yscale('log')
ax.set_title('log')
ax.grid(True)
plt.show()

#%%
# 箱型图
# Random test data
np.random.seed(123)
all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# rectangular box plot
bplot1 = axes[0].boxplot(all_data,vert=True,patch_artist=True)  
# notch shape box plot
bplot2 = axes[1].boxplot(all_data,notch=True,vert=True,patch_artist=True)
# fill with colors
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(all_data))], )
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')
# add x-tick labels
plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],
         xticklabels=['x1', 'x2', 'x3', 'x4'])
plt.show()

#%%
# 积累分布
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
np.random.seed(0)
mu = 200
sigma = 25
n_bins = 50
x = np.random.normal(mu, sigma, size=100)
fig, ax = plt.subplots(figsize=(8, 4))
# plot the cumulative histogram
n, bins, patches = ax.hist(x, n_bins, normed=1, histtype='step',
                           cumulative=True, label='Empirical')
# Add a line showing the expected distribution.
y = mlab.normpdf(bins, mu, sigma).cumsum()
y /= y[-1]
ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')
# Overlay a reversed cumulative histogram.
ax.hist(x, bins=bins, normed=1, histtype='step', cumulative=-1,
        label='Reversed emp.')
# tidy up the figure
ax.grid(True)
ax.legend(loc='right')
ax.set_title('Cumulative step histograms')
ax.set_xlabel('Annual rainfall (mm)')
ax.set_ylabel('Likelihood of occurrence')
plt.show()
#%%
#直方图
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
np.random.seed(0)
# example data
mu = 100  
sigma = 15 
x = mu + sigma * np.random.randn(437)
num_bins = 50
fig, ax = plt.subplots()
# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, normed=1)
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
ax.plot(bins, y, '--')
ax.set_xlabel('Smarts')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

#%%
# Pandas的数据框和序列
import os
print(os.getcwd())
os.chdir('C:/Users/XIEW0/Desktop/Data Science with Python\第三课')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('direct_input_data.csv',index_col='Date')
df.index = pd.to_datetime(df.index)
fig, ax = plt.subplots(1, 1)
ax.plot(df.index[0:200],df['AAPL'][0:200])
ax.set_ylabel(u'AAPL',fontproperties='SimHei')
ax.set_xlabel(u'时间',fontproperties='SimHei')
ax.grid(True)
import matplotlib.dates as mdates
xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
ax.xaxis.set_major_formatter(xfmt)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
fig.autofmt_xdate()
plt.show()

#%%
#股票的蜡烛图
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()
# download dataframe
df = pdr.get_data_yahoo("AAPL", start="2018-04-01", end="2018-07-30")

import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
fig, ax = plt.subplots()
df['Date']=df.index
df['Date'] = df.index.map(mdates.date2num)
quotes = df[['Date','Open','High','Low','Close']]
candlestick_ohlc(ax, quotes.values, width=0.6,
                 colorup='green', colordown='red')
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
fig.autofmt_xdate()
ax.set_ylabel(u'价格',fontproperties='SimHei')
ax.set_xlabel(u'时间',fontproperties='SimHei')
ax.set_title(u'苹果公司股价走势图',fontproperties='SimHei')
ax.grid(True)
plt.show()

#%%