# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:48:03 2016

@author: garam
"""

from matplotlib import pyplot as plt
import scipy as sp
stock_price_today = 9.15 # stock price at time zero
T=1.                     # maturity date (in years)
n_steps=100            # number of steps
mu =0.15                 # expected annual return
sigma = 0.2              # volatility (annualized)
sp.random.seed(1234)     # seed()
n_simulation =5          # number of simulations

dt = T/n_steps
S = sp.zeros([n_steps], dtype=float)
x=range(0, int(n_steps), 1)
for j in range(0, n_simulation):
    S[0]= stock_price_today
    for i in x[:-1]:
        e=sp.random.normal()
        S[i+1]=S[i] + S[i]*(mu-0.5*pow(sigma, 2))*dt + sigma*S[i]*sp.sqrt(dt)*e;
    plt.plot(x, S)    

plt.figtext(0.2,0.8,'S0='+str(S0)+',mu='+str(mu)+',sigma='+str(sigma))
plt.figtext(0.2,0.76,'T='+str(T)+', steps='+str(int(n_steps)))
plt.title('Stock price (number of simulations = %d ' % n_simulation +')')
plt.xlabel('Total number of steps ='+str(int(n_steps)))
plt.ylabel('stock price')
plt.show()

