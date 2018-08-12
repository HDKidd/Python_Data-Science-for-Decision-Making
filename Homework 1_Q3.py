#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:40:39 2018

@author 何德坤 He Dekun
Data Science for Decision Making II: Homework 1
Question 3: Fibonacci Sequence
"""

#%%
# Solution 

# 自定义递归函数：

def Fibo(n):
    if not isinstance(n, int):
        raise TypeError('Input must be integer')    # 参数检查，规定为整数
    if n <= 0:
        raise TypeError('Input must be positive')    # 参数检查，规定为正数 
    if n == 1 or n == 2:
        return 1             # 输入是 1 或者 2 时，根据 fibo 的定义，返回 1
    else:
        return Fibo(n-1) + Fibo(n-2)    # 输入大于 2 时，返回 f(n-1)+f(n-1)， 递归
    
# 输入项数 x，输出斐波那契数列的第 x 项的值：
        
x = int(input("Please insert a positive integer: ",))    # 将输入转化为整数
print ("Fibonacci Sequence value, of term %d, is: " % x, Fibo(x))

# 扩展：输出斐波那契数列：

fibo_list = []
for i in range(1, x+1):
    fibo_list.append(Fibo(i))    # 将 fibo 数从小到大填充到空的 list 中
print ("Fibonacci Sequence, to term %d, is: " % x, fibo_list)

#%%
# Output:
# Please insert a positive integer: 10
# Fibonacci Sequence Value is:  55
# Fibonacci Sequence, to term 10, is:  [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

#%%
# 总结：
"""
1)  fibo 数列函数的关键是实现递归，在自定义函数时，返回的内容设为函数本身（参数不同）
    从而实现递归；
3)  上面定义 Fibo 函数的时候，也可以令 n <= 0 时 return 0, 这样既可避免非正整数无解，
    也不需写参数检查，函数更为简洁；
2)  除递归算法外，还可以用循环方法实现，相较于循环方法，递归算法在项数 x 较大时求解
    速度较慢， 不过循环方法的函数没有递归方法来的简洁易读。
"""