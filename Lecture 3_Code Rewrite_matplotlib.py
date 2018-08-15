#!/usr/bin/env python 3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:15:21 2018

@author: He Dekun 
Rewrite code of Lecture 3_Matplotilb
"""

#%%
# # Matplotlib图表绘制
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
os.chdir('D:/03 NTU.MSF/01 Courses/AY1718T4/AY1718T4-FF6120-Data Science for Decision Making II/01 Lecture Notes/Part I_Python/fileopen')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('pandas_out_data.csv',)
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