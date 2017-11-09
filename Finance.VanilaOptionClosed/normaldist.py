# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 02:31:48 2016

@author: garam
"""

import numpy as np
from scipy import exp, sqrt, stats
from matplotlib import pyplot as plt
z=0.325
def f(t):
    return stats.norm.pdf(t)
plt.ylim(0.,0.45)
x=np.arange(-3,3,0.1)
y1=f(x)
plt.plot(x,y1)
x2=np.arange(-4,z,1/40.)
sum=0
delta=0.05
s=np.arange(-10,z,delta)
for i in s:
    sum+=f(i)*delta
plt.annotate('area is '+str(round(sum,4)),xy=(-1,0.25),xytext=(-3.8,0.4), arrowprops=dict(facecolor='red',shrink=0.01))
plt.annotate('z= '+str(z),xy=(z,0.01))
plt.fill_between(x2,f(x2))
