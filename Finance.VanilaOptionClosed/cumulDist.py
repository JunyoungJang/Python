# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 02:40:06 2016

@author: garam
"""
import numpy as np
from scipy import exp, sqrt, stats
from matplotlib import pyplot as plt
z=0.325
def f(x):
    return stats.norm.cdf(x)
x=np.arange(-3,3,0.1)
y1=f(x)
y2=np.ones(len(x))*0.5
x3=[0,0]
y3=[0,1]
plt.plot(x,y1)
plt.plot(x, y2, 'b-')
plt.plot(x3,y3)
plt.annotate('f(z)=f('+str(z)+') is '+str(np.round(f(z),4)),xy=(z,f(z)),xytext=(z-3,f(z)), arrowprops=dict(facecolor='red',shrink=0.01))
plt.annotate('z is '+str(z),xy=(z,0),xytext=(1.5,0.3), arrowprops=dict(facecolor='blue',shrink=0.01))
