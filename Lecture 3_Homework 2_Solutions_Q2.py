#!/usr/bin/env python 3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:07:07 2018

@author 何德坤 He Dekun
Data Science for Decision Making II: Homework 2
Question 2: 美国股票作图
"""

#%% 导入库

import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('D:/03 NTU.MSF/01 Courses/AY1718T4/AY1718T4-FF6120-Data Science for Decision Making II/02 Homework/HW2_Data')    # 修改环境变量

#%% 1. 读取 EPS 数据并存入数据框  2. 计算均值与方差

df_eps = pd.read_csv('HW2_Q2_EPS.csv', index_col='Year')    # 读入数据

mu = df_eps.mean()    # 计算均值
var = df_eps.var()    # 计算方差

#%% 制作折线图（合并图）

# 画布创建与设定大小
fig_plot, ax_plot = plt.subplots(figsize=(15, 6))

# 作折线图图（三条线画在同一个画布上）
for stk in ['FB', 'AMZN', 'AAPL']:
    x = df_eps.index                                      # 设定x值
    y = df_eps[stk].values                                # 设定y值
    plt.plot(x, y, 'o-', linewidth=2,  markersize=8)      # 作图

    # 设定节点标签
    for labelx, labely in zip(x, y):
        plt.text(labelx, labely, '%.2f' % labely, ha='left', va= 'bottom',fontsize=12)

# 设定 参考线和 label
ax_plot.xaxis.grid(True, which='major', linestyle='--', color='lightgrey')    # x轴参考线
ax_plot.yaxis.grid(True, which='major', linestyle='--', color='lightgrey')    # y轴参考线
ax_plot.set_axisbelow(True)                                                   # 参考线置底
ax_plot.set_title('EPS Plot', pad=20, fontsize=25)                            # 标题
plt.legend(['FB', 'AMZN', 'AAPL'], loc = 'upper left', fontsize=15)           # 图例

# 设定 坐标轴细节
ax_plot.set_xlabel('Year', labelpad=10, fontsize=15)                          # x轴标记
ax_plot.set_ylabel('EPS', labelpad=10, fontsize=15)                           # y轴标记
ax_plot.set_xticks(x)                                                         # x轴刻度
ax_plot.set_yticks(range(-2,12,2))                                            # y轴刻度

# 输出和保存图像
plt.show()
plt.savefig('HW2_Q2_EPSPlot.jpg')

#%% 制作折线图（子图）

# 画布创建与设定 (一行三个子图,共享x轴)
fig_plots, ax_plots = plt.subplots(3,1, figsize=(25,9), sharex=True)

# 预备作图参数
x = df_eps.index                    # 设定 x值
color = ['blue', 'red', 'green']    # 设定子图线条颜色列表
num = 0                             # 子图索引号

# 作折线图图（三条线分别画在三个子图上）
for stk in ['FB', 'AMZN', 'AAPL']:
    y = df_eps[stk].values                                                              # 设定 y值
    ax_plots[num].plot(x, y, 'o-', linewidth=2,  markersize=10, color=color[num])       # 作图

    # 设定节点标签
    for labelx, labely in zip(x, y):
        ax_plots[num].text(labelx, labely, '%.2f' % labely, ha='center', va= 'bottom', fontsize=12)

    # 设定 坐标轴，参考线，label 等部件
    ax_plots[num].xaxis.grid(True, which='major', linestyle='--', color='lightgrey')    # x轴参考线
    ax_plots[num].yaxis.grid(True, which='major', linestyle='--', color='lightgrey')    # y轴参考线
    ax_plots[num].set_axisbelow(True)                                                   # 参考线置底
    ax_plots[num].set_ylabel('EPS', labelpad=10, fontsize=15)                           # y轴标记
    ax_plots[num].set_yticks(range(-2,12,2))                                            # y轴刻度
    ax_plots[num].legend([stk], loc = 'upper left', fontsize=15)                        # 图例

    num = num + 1                                                                       # 子图索引步进

# 设定总图部件
ax_plots[0].set_title('EPS Plots', pad=20, fontsize=25)                                 # 标题
ax_plots[2].set_xlabel('Year', labelpad=10, fontsize=15)                                # x轴标记
ax_plots[2].set_xticks(df_eps.index)                                                    # x轴刻度

# 输出和保存图像
plt.show()
plt.savefig('HW2_Q2_EPSPlots.jpg')
