#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:17:08 2018

@author 何德坤 He Dekun
Data Science for Decision Making II: Homework 1
Question 2: Festival Dictionary 
"""

#%%
# Solution 

# 字典创建和设定:

holiday = {}    # 创建空字典

for n in range(180801, 180832):    # 遍历2018年8月的所有日期，根据规则自动填充进字典
    if n == 180809:
        holiday[n] = "1 National Day"    # 节假日日期的 key，value 为 1 
    elif n == 180822:
        holiday[n] = "1 Hari Raya Naji"
    else: 
        holiday[n] = "0 Not holiday"    # 除了节假日以外的日期 key，value 均为 "0 Not holiday"

# 输入和输出交互：       
while True:
    check_date = int(input("Please insert a date in Aug 2018 (format: 180801): ",))    
    if check_date not in holiday:    # 输入格式和范围检查
        print("Wrong format or Not in Aug 2018, please reinsert: ")
    else:
        print(holiday[check_date])    # 对符合要求的输入，输出对应 key 的 value
        break
#%%
# Output:
# Please insert a date in Aug 2018 (format: 180801): 180101
# Wrong format or Not in Aug 2018, please reinsert: 
        
# Please insert a date in Aug 2018 (format: 180801): 180809
# 1 National Day
        
#%%
# 总结：
"""
1)  创建空字典比直接创建包含所有键值的字典更便捷和灵活，使用循环语句自动填充字典即可；
2)  输入的格式默认为字符串，需转化为与字典的 key 一致的数据类型，方可检索成功；
3)  此方法的缺点是每换一个月份就要重新定义一遍字典。
"""
