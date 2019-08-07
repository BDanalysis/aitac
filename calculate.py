import sys
import math  
import random
import datetime
import numpy as np
import pandas as pd
import warnings
from numpy import *
from numba import njit
from scipy import special
from scipy.stats import norm
from scipy.stats import rv_continuous, gamma
from sklearn import preprocessing

calFile = "calculateFile_abCN.txt"
totalmeanFile = "calculateMeanFile.txt"
cal = open(calFile,'r')
tot = open(totalmeanFile,'w')
total_CN = []
count = 0

for i in range(0,9):
    total_CN.append(0)

for j in cal.readlines():
    count += 1
    c_array = j.split("\t")
    for k in range(0,9):
        total_CN[k] = total_CN[k] + float(c_array[k])

for i in range(0,9):
    total_CN[i] = total_CN[i]/count

for i in range(0,9):
    tot.write(str(total_CN[i]) + "\n")

cal.close()
tot.close()



    

