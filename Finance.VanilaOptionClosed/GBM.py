# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:14:15 2016

@author: garam
"""

from time import time
from math import exp, sqrt, log
from random import gauss, seed

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt




S0=3
r=0.02
sigma = 0.2
T=3.0
I = 100
M = 100
dt = T / M
S = np.zeros((M + 1, I))
S[0] = S0
for t in range(1, M + 1):
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt 
            + sigma * np.sqrt(dt) * npr.standard_normal(I))
            
plt.plot(S[:, :10], lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.grid(True)