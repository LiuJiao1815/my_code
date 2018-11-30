##导入函数库

import numpy as np 
import pandas as pd 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import time 
import scipy.optimize as opt 
import scipy as sp 
import math 
#import seaborn     #一种作图软件
from matplotlib import cm 
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import CubicSpline 
from scipy.stats import norm, poisson
from scipy.optimize import leastsq
import xgb
#numpy 是一种非常好用的数据包，可以用来生成很多的数组
'''
a = np.arange(0,60,10).reshape(-1,1) + np.arange(6)
print(a)

# 1 创建列表
L = [1,2,3,4,5,6]
print(L)
a = np.array(L)
print(a)
b= np.array([1,2,3,4,5,6])
print(b)
#总结数组和列表的区别为列表有逗号，数组输出后没有逗号。
#创建多维数组
b = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=np.float)
print(b)
print(b.dtype)
'''

#2、使用弄函数创建
#如果生成一定规则的数据，可以使用Numpy 提供的专门函数
#arange函数类似于python的range函数：制定起始值、终止值和步长来创建数组
#和python的range类似，arange同样不包括终值；但 arange可以生成浮点类型，而range只能是整点类型
np.set_printoptions(linewidth=100, suppress=True)   #以小数的形式表示
a = np.arange(1,10,0.5)
#print(a)

a = np.arange(1, 20, 3, dtype = float)
#print(a)

# linspace 函数通过制定起始值、终止值和元素个数来创建数组，缺省包括终止值
b = np.linspace(1,10,10)
print('b=',b)