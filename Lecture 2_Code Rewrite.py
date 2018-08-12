#!/usr/bin/env python 3
# -*- coding: utf-8 -*-
"""
@author: He Dekun 
Rewrigt code of Leture 2_Python Basic
"""
#%%
#Lesson 2
#Python的基础

#%%
# 第一个示例程序
print ("Hello World! 这是我第一次Python尝试")

#%%
# 数字数据类型
a = 2 #整数
b = 0.2245 #浮点数
c = True #布尔值
type(a)
type(b)
type(c)

#%%
# 算数操作符
print (2+3)  
print (3-2)
print (3*2)
print (9/3)
print (10%3)
print (4**3)
print (10//3)

#%%
# 赋值操作符
a = 6
print (a)
a+=2 # a = a + 2
print(a)
a-=2 # a = a - 2
print(a)
a*=2 # a = a * 2
print(a)
a/=2 # a = a / 2
print(a)
a%=2 # a = a % 2
print(a)
a**=2 # a = a ** 2
print(a)
a//=2 # a = a // 2
print(a)

#%%
# 比较操作符
print (3 == 2)
print (3 != 2)
print (3 > 2)
print (3 < 2)
print (3 >= 2)
print (3 <= 2)

#%%
# 逻辑操作符
print ((3>2) and (4>3))
print ((3>2) and (4<3))
print ((3>2) or (4>3))
print ((3<2) and (4>3))
print (not (3>2))

#%%
# 操作优先级
# 幂 **
# 乘， 除， 取模， 取商 * / % //
# 加， 减 + -
# 比较操作符 
# 赋值操作符
# 逻辑操作符
10 > 3**2 + 3 and 5 < 4 % 3

#%%
# 变量与赋值
apple_price = 170
print (apple_price)
amazon_price = 1500
google_price = 1200
index = amazon_price + google_price
print (index)

#%%
# 布尔表达式
print (True == 1)
print (True+2)
print ((1<3)*10)

#%%
#if 条件选择
#example1:
a = 10
if a > 3:
    print (3+2)
elif a > 2:
    print (3*2)
else:
    print(3-2)
#example2:
MA_20 = 60.33
MA_60 = 70.10
if MA_20 > MA_60:
    signal = 1
elif MA_20 < MA_60:
    signal = -1
print (signal)

#%%
#while循环
#example1:
a = 1
b = 1
while a < 5:
    a+=1
    b-=1
print (b)
#example2:
MA_20 = 60.33
MA_60 = 70.10
number_converge_day = 0;
while MA_20 < MA_60:
    MA_20+=1
    MA_60-=1
    number_converge_day +=1
print(number_converge_day)

#%%
#for 循环
#example1:
for idx in range(2,6):
    print(idx)
#example2:
for idx in range(2,10):
    if idx == 5:
        continue
    if idx == 7:
        break
    print(idx)

#%%
# 数据结构
# 标量：整数 浮点数 布尔值
# 序列：列表，字符串，元组，Unicode字符串，字节数组，缓冲区和xrange对象
# 列表
List1 = [ 2 , 0.782323 , True]
print(List1[0])
print(List1[-1])
print(List1[0:2])
print(List1[:2])
print(List1[0:])
print(List1[:])        
List1.append(False)
print(List1)
List2 = [3 , 4]
List1.extend(List2)
print(List1)
List1.insert(3 , 'NTU')
print(List1)
List1.remove('NTU')
print(List1)
List1.index(True)
List1.count(False)

#%%
#字符串
str1 = 'learn python'
print (str1)
print (str1[0])
print (str1[-1])
print(str1[:7])
str1 = str1 + str1
print(str1)
str1.find('rn')
str1.split()
str1.count('ea')
str1.replace('ea','')

#%%
#unicode字符串
unicode_str = u'\u4f60\u597d'
print(unicode_str)
#元组
tuple1 = (2 , 0.782323 , True)
tuple(list(tuple1))

#%%
# 映射：字典
category = {'apple':'fruit','Zootopia':'film'\
            ,'football':'sport'}
print(category['apple'])
category['lemon']='fruit'
print(category)
del category['lemon']
print(category)
keys = category.keys()
print (keys) 

#%%
# 集合
set1 = {1,2,3}
set2 = {2,4,6}
print(set1)
print(set1 - set2)
print(set1 | set2)
print(set1 & set2)
print(set1 ^ set2)
print(set1 < set2)
print(set1 > set2)

#%%
# 文件的读写
import os
print(os.getcwd())
os.chdir('D:/03 NTU.MSF/01 Courses/AY1718T4/AY1718T4-FF6120-Data Science for Decision Making II/01 Lecture Notes/Part I_Python/fileopen')
print(os.getcwd())
#example 1
file_handler=open('fileopentest.txt', mode = 'r')
a = file_handler.readlines()
data=[]
for line in a:
    line = line.strip()
    data_line = line.split(",")
    data.append(data_line)
file_handler.close()
#example 2
file_handler=open('fileopentest.txt', mode = 'r')
b = file_handler.readlines()
file_handler.close()
#example 3
f = open('fileopentest.txt','w')
data=[['1','2'],['3','4']]
line1 = ','.join(data[0])
f.write(line1+'\n')
line2 =','.join(data[1])
f.write(line2+'\n')
f.close()
 

#%%
# 函数的定义和使用
#example 1
def sum_number (a,b):
    return a+b
print(sum_number(1,4))
#example 2
def moving_average (a , n):
    ma = [];
    for i in range(0,len(a)):
        if i+1<n:
            ma.append(0)
        else:
            ma.append(sum(a[i-n+1:i+1])/n)
    return ma
stock_price = [9.2,9.3,9.5,9.0,9.8,9.2,9.9,10.5]
print(moving_average(stock_price,2))

#%%
# lambda 语句
# exmaple 1
g = lambda a,b : a+b
print(g(1,4))
# exmaple 2
from math import log
def make_log_function(base):
    return lambda x:log(x,base)
My_LF = make_log_function(3)
print (My_LF(9))

#%%
# 函数参数
#位置
def sum_number (a,b,c):
    return a+b
print(sum_number(1,4,3))
print(sum_number(a=1,c=4,b=3))
#任意数量参数
def sum_number (*a):
    return sum(a)
print(sum_number(1,4,3))

#%%




