#!/usr/bin/env python 3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 20:41:21 2018

@author: He Dekun 

Debug for DataFrame.sort_values Error:
    ValueError: The column label 'APPL' is not unique.

"""

#%% 
# original code:

import numpy as np
import pandas as pd

a = np.random.standard_normal([9, 4])    
df = pd.DataFrame(a)    
df.columns = [['APPL', 'GOOG', 'FB' , 'AMZN']]    # Bug: Wrongly setting columns into array instead of list 
print (df)

df.sort_values('APPL', ascending=False)

#%%
# Error Description:
# ValueError: The column label 'APPL' is not unique.
# For a multi-index, the label must be a tuple with elements corresponding to each level.

# Reason:
# df.columns = [[...]] making columns as array, columns are showed as ('APPL',) in Variable explorer
# when calling sort_values by using one string lable 'APPL', raise error

#%%
# right code:

a = np.random.standard_normal([9, 4])    
df = pd.DataFrame(a)    
df.columns = ['APPL', 'GOOG', 'FB' , 'AMZN']     
print (df)

df.sort_values('APPL', ascending=False)

# output: OK; 
# columns are showed as APPL, OK