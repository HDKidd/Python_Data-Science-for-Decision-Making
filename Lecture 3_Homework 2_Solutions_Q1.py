#!/usr/bin/env python 3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:00:51 2018

@author 何德坤 He Dekun
Data Science for Decision Making II: Homework 2
Question 1: 中国股票作图
"""

#%% 导入库

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc as can
import os
# pd.core.common.is_list_like = pd.api.types.is_list_like    # 若出现 importError：cannot import name 'is_list_like' 则运行此行
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf    # yahoo 修改了 API 导致 pdr 0.6.0 和之前的版本都接不上，新版本出来之前用此包修正

yf.pdr_override()    # 用 fix_yahoo_finance 打补丁

#%% 1.从 hahoo 获取数据； 2.计算日收益率； 3.汇总并输出到 csv 文件

os.chdir('D:/03 NTU.MSF/01 Courses/AY1718T4/AY1718T4-FF6120-Data Science for Decision Making II/02 Homework/HW2_Data')    # 修改环境变量

df_pricevol = pd.DataFrame()    # 创建空数据框
df_logreturn = pd.DataFrame()    # 创建空数据框

for stock_code in range(600027,600032):
    df = pdr.get_data_yahoo('%s.SS' % stock_code , start='2017-08-15', end='2018-08-14')    # 批量下载股票价量数据
    df.to_csv('%s.SS.csv' % stock_code)    # 批量输出为 csv 文件

    # 把各个单表汇总到一个数据框中：
    for item in ['Open', 'High', 'Low', 'Close']:
        df_pricevol['%s_%s' % (stock_code, item)] = df['%s' % item]

    # 计算每只股票的日对数收益率，汇总到数据框中：
    df_logreturn['%s_logreturn' % stock_code] = np.log(df_pricevol['%s_Close' % stock_code] / df_pricevol['%s_Close' % stock_code].shift(1))

df_pricevol.to_csv('Pricevol Data.csv')    # 输出价量汇总表的 csv 文件
df_logreturn.to_csv('Logreturn Data.csv')    # 输出日对数收益率汇总表的 csv 文件

#%% 制作箱型图

fig_bp, ax_bp = plt.subplots(figsize=(15, 6))    # 创建图框，设定大小
plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=0.2)    # 调整画布大小（以左下角为坐标原点）

data_bp = df_logreturn.dropna(0,'all')    # 对 logreturn 数据框进行 NaN 值删除, 预备作图数据

# 制作箱型图：参数（数据，凹口，离群点形状，横置图形，箱体宽度，Line2D 或 Patch箱体形态，label）
bp = plt.boxplot(data_bp.values, notch=1, sym='o', vert=0, widths=0.6, patch_artist=1, labels=df_logreturn.columns)

# 设定各个部件的图画属性
plt.setp(bp['boxes'], color='mediumslateblue')    # 箱体
boxColors = ['darkkhaki', 'royalblue']    # 箱体颜色
plt.setp(bp['whiskers'], color='red')    # 突线
plt.setp(bp['fliers'], color='black')    # 离群值
ax_bp.xaxis.grid(True, which='major', linestyle='--', color='lightgrey')    # 横轴参考线
ax_bp.set_axisbelow(True)    # 参考线置底

# 设定 label
ax_bp.set_title('Boxplots of Daily Return', pad=20, fontsize=25)
ax_bp.set_xlabel('Daily LogReturn', labelpad=10, fontsize=15)
ax_bp.set_ylabel('Stocks', labelpad=10, fontsize=15)

# 输出和保存图像
plt.show()
plt.savefig('HW2_Q1_Boxplot.jpg')

#%% 制作直方图

# 画布创建与设定
fig_hist, ax_hist = plt.subplots(figsize=(15, 6))    # 创建图框, 设定大小
plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=0.2)    # 调整画布大小

# 预备作图数据：
data_hist = df_logreturn.dropna(0,'all')    # NaN 值删除
data_hist = data_hist['600027_logreturn']    # 选取 600027 这只股票
mu = data_hist.mean()    # 取均值
sigma = data_hist.std()    # 取标准差

# 制作直方图：参数（数据，颜色，方块数量）
n, bins, patches = plt.hist(data_hist.values, color='darkviolet', bins=60)

# 制作正态分布拟合曲线
fitline = mlab.normpdf(bins, mu, sigma)
fig_hist, ax_hist.plot(bins, fitline, '--')

# 设定 参考线和 label
ax_hist.yaxis.grid(True, which='major', linestyle='--', color='lightgrey')    # y轴参考线
ax_hist.set_axisbelow(True)    # 参考线置底
ax_hist.set_title('Histogram of Daily Return: 600027', pad=20, fontsize=25)
ax_hist.set_xlabel('Daily LogReturn', labelpad=10, fontsize=15)
ax_hist.set_ylabel('Counts', labelpad=10, fontsize=15)
plt.legend(['Fit Line', 'Histogram'], loc = 'upper right', fontsize=15)    # 图例

# 输出和保存图像
plt.show()
plt.savefig('HW2_Q1_Histogram.jpg')

#%% 制作蜡烛图

# 画布创建与设定
fig_cand, ax_cand = plt.subplots(figsize=(15, 6))    # 创建图框, 设定大小
plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=0.2)    # 调整画布大小

# 预备作图数据：
df_pricevol['Date'] = df_pricevol.index
df_pricevol['Date'] = df_pricevol.index.map(mdates.date2num)    # 由于蜡烛图作图时不能引入日期型参数，故先转化为数值
stk = 600027    # 选择股票
data_cand = df_pricevol[['Date', '%s_Open' % stk, '%s_High' % stk, '%s_Low' % stk, '%s_Close' % stk]]

# 制作蜡烛图
candle = can(ax_cand, data_cand.values, width=0.6, colorup='red', colordown='green')

# 设定坐标轴和参考线
ax_cand.yaxis.set_major_locator(plt.MaxNLocator(11))    # 设定 y轴最大坐标数
ax_cand.xaxis.set_major_locator(plt.MaxNLocator(20))    # 设定 x轴最大坐标数
ax_cand.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))    # 设定 x轴坐标格式
ax_cand.yaxis.grid(True, which='major', linestyle='--', color='lightgrey')    # y轴参考线
ax_cand.xaxis.grid(True, which='major', linestyle='--', color='lightgrey')    # x轴参考线
ax_cand.set_axisbelow(True)    # 参考线置底

# 设定 label
ax_cand.set_title('Candlestick of Stock: %s' % stk, pad=20, fontsize=25)
ax_cand.set_xlabel('Date', labelpad=10, fontsize=15)
ax_cand.set_ylabel('Price (JPY)', labelpad=10, fontsize=15)

# 输出和保存图像
plt.show()
plt.savefig('HW2_Q1_Candlestick .jpg')
