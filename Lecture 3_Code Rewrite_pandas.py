#!/usr/bin/env python 3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:28:21 2018

@author: He Dekun 
Rewrite code of Leture 3_Pandas
"""

#%%
# 数据框 DataFrame

import numpy as np
import pandas as pd

data = {'id':['Jack', 'Sarah', 'Mike'], 'age':[18, 35, 20], 'cash':[10.53, 500.7, 13.6]}    # 创建一个字典
df1 = pd.DataFrame(data)    # 调用数据框构造函数并将返回值赋值给 df

print (df1)    # 输出的是默认格式的数据框,会对变量索引自动排序

df2 = pd.DataFrame(data, columns=['id', 'age', 'cash'], index=['one', 'two', 'three'])
# index 指第一列索引序列，如不设定参数，则默认使用0,1,2...，columns 指第一行索引序列，规定了其排序

print (df2)    # 输出的是指定索引格式的数据框
print (df2['id'])    # 输出 变量名为 'id' 的数据值，实际上是输出一个序列

#%%
# 数据框：时间序列

a = np.random.standard_normal([9, 4])    # 使用 numpy 模块创建随机标准正态 9x4 数组
a.round(6)    # 设定小数位数为 6 位
df = pd.DataFrame(a)    # 用数组 a 创建 DataFrame
df.columns = ['APPL', 'GOOG', 'FB' , 'AMZN']    # 设定 DataFrame 的列索引（即变量名）
print (df)

dates = pd.date_range(start='2015-1-1', periods=9, freq='M')    # 详细参数参考 doc，M: month end frequency
print (dates)
df.index = dates    # 设定 DataFrmae 的行索引（时间）
print (df)

dates = pd.date_range(start='2015-1-1', periods=200, freq='B')    # B: business day frequency
print (dates)    # 输出的结果会省略中间数据

dates = pd.date_range(start='2015-1-1', periods=200, freq='H')    # H: hourly frequency
print (dates)  

#%%
# 数据框基本访问

print (df.dtypes)        # 返回数据类型
print (df.head(6))       # 返回顶部数据，参数不输则默认为 5
print (df.tail(3))       # 返回底部数据，参数不输则默认为 5
print (df.index)         # 返回 index 索引序列
print (df.columns)       # 返回 columns 索引序列
print (df.values)        # 返回 value 数组
print (df.describe())    # 返回一般的描述性统计量
print (df.sum())         # 返回每个 column 的求和
print (df.mean())        # 返回每个 column 的均值
print (df.cumsum())      # 返回每个 column 的累计求和
print (np.sqrt(df))      # 对每个可以求平方根的函数求平方根（此函数不能通过.函数的格式调用）

# 切片访问
print(df[0:3])                          # 返回索引0到索引3的值
print(df['2015-01-31':'2015-05-31'])    # 返回 index 切片内的值
print(df['2015-01-31':'2015-05-15'])    # 返回 index 切片内的值
print(df['20150131':'20150515'])        # 返回 index 切片内的值（自动识别不同格式的索引）
print(df['2015/01/31':'2015/05/15'])    # 返回 index 切片内的值（自动识别不同格式的索引）

print(df.loc['2015/01/31':'2015/05/15',['APPL', 'AMZN']])    # 返回 index 切片和 columns 切片内的值
print(df.iloc[1, 1])    # 返回索引坐标为[1,1]的值
print(df.iloc[1:3, 1:3])    # 返回索引切片内的值
print(df[df.APPL>0])    # 返回只显示满足条件的值的完整数据框，不满足数据的值显示为NaN

#%%
# 排序
df.sort_index(axis=0, ascending=False)    # 对索引进行排序， axis=0 为 index 索引
df.sort_index(axis=1, ascending=False)    # 对索引进行排序， axis=1 为 columns 索引

df.sort_values('APPL', ascending=False)   # 
print (df)
pd.DataFrame.sort_values(df, 'APPL')      # 

df['APPL']
print (type(np.array(df['APPL'])))

# 从数据框中删除系列
del df['APPL']
print (df)

# 从数据框中新增系列
df['APPL'] = df['AMZN'] > 0
print (df)

#%%
# 系列 Series

s = pd.Series({'a':4, 'b':9, 'c':16}, name='number')    # 创建系列
print (s)
print (s[0])    # 返回系列元素
print (s[:3])    # 返回系列切片内的元素
print (s['b'])    # 按索引返回元素值
s['b'] = 25    # 按索引修改元素值
print (s)

df_new = df['GOOG']    # 从数据框中创建系列
type (df_new)
print (df_new)

#%%
# 文件读写 (循环法，不便捷)
import os

rows = 5000
a = np.random.standard_normal((rows, 5))    # 创建一个正态随机的 rows x 5 矩阵
print(a)
a.round (4)    # 取4位小数
t = pd.date_range(start='20140101', periods=rows, freq='H')    # 创建日期索引

print (os.getcwd())    # 获取环境变量值
os.chdir('D:/03 NTU.MSF/01 Courses/AY1718T4/AY1718T4-FF6120-Data Science for Decision Making II/01 Lecture Notes/Part I_Python/fileopen')
path = os.getcwd()    # 设定文件读写路径

csv_file = open ('out_data.csv', 'w')    # 创建并打开文件
header = 'date, APPL, GOOG, FB, AMZN, MSFT, \n'
csv_file.write(header)    # 写入 header

for t_variable, (APPL_v, GOOG_v, FB_v, AMZN_v, MSFT_v) in zip (t, a): 

# zip zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

    s = '%s, %f, %f, %f, %f, %f\n' % (t_variable, APPL_v, GOOG_v, FB_v, AMZN_v, MSFT_v)
    csv_file.write(s)    # 写入值

csv_file.close()    # 关闭文件

csv_file = open ('out_data.csv', 'r')    # 读出标题
for i in range(5):
    print (csv_file.readline())

csv_file = open ('out_data.csv', 'r')    # 读出内容
content = csv_file.readlines()

csv_file.close()
#%% 
# 文件读写（pandas法，便捷）

df = pd.DataFrame(a)    # 创建一个数据框
df.columns = ['APPL', 'GOOG', 'FB', 'AMZN', 'MSFT']    # 定义数据框的变量名
df.index = t    # 定义数据框的索引列
df.to_csv('pandas_out_data.csv')    # 输出到 csv 文件
df_read = pd.read_csv('pandas_out_data.csv', index_col=None, parse_dates = True)   
# 从 csv文件读入。参数：文件名（地址），index_col 指向 index 的列标题，无则None， parse_dates 是指识别为日期格式 

print (df_read)

