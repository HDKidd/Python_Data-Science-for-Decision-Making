#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Created on Sat Aug 11 19:21:33 2018

@author 何德坤 He Dekun
Data Science for Decision Making II: Homework 1
Question 1: Bubble Sort Algorithm
"""

#%%
# Solution 1：Use while

SList = [5, 6, 3, 4, 8, 1, 9, 0, 2]    # 待排序 list

n = 0
while n+1 < len(SList):    # 控制循环不超出 list 边界
    if SList[n] > SList[n+1]:
        SList[n], SList[n+1] = SList[n+1], SList[n]    # 对换元素
        n = 0    # 发生对换后重置循环起始位置，以重新遍历序列
        continue
    else:
        n = n+1    

print ("Sorting Success! Ascending order is: ", SList)

#%%
# Output:
# Sorting Success! Ascending order is:  [0, 1, 2, 3, 4, 5, 6, 8, 9]

#%%
# Solution 2：Use for

SList = [5, 6, 3, 4, 8, 1, 9, 0, 2]    # 待排序 list

for n in range(0, len(SList)):    
    for m in range(n+1, len(SList)):
        if SList[n] > SList[m]:
            SList[n], SList[m] = SList[m], SList[n]    # 对换元素

print ("Sorting Success! Ascending order is: ", SList)

#%%
# Output:
# Sorting Success! Ascending order is:  [0, 1, 2, 3, 4, 5, 6, 8, 9]

#%%
# 总结：两种方法的比较
"""
1)  使用 while 方法的代码相对较长，阅读性较差；而 for 方法的代码简洁易读，逻辑更清晰。
2)  从运算效率上看，while 循环视原来序列的顺序不同，而需要不同的循环次数，当序列本身为
    顺序时，只需要循环 l-1 次(l为序列长度）。但当序列本身为完全逆序时，则需要循环的次
    数为：1 + (2+1) + (3+2+1) + ... + (l-1+l-2+...+1) + l-1 。
    而 for 方法无论原始序列如何，都要循环 (l-1)*l/2 次。
    可见 while 方法的运算效率不稳定，有时会比 for 方法快，有时比 for 方法慢，且当序列
    长度非常大时，最大循环次数很高，运算效率显著降低。
    相对来说，for 的运算效率是稳定且可以接受的。
"""