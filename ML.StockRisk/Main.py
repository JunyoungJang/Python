from __future__ import division, print_function, absolute_import
import pandas as pd
import numpy as np
from loadmodel import Estimation_Index_updown, RiskZone_Criterion

def Result1(result):
    index = result.index(max(result))
    prob = max(result)
    if index == 0:
        str = 'Normal'
    elif index == 1:
        str = 'Risk1'
    elif index == 2:
        str = 'Risk2'
    return str, index, prob

def Result2(result):
    index = np.argmax(result)
    prob = max(result)
    if index == 0:
        str = 'Decrease'
    elif index == 1:
        str = 'Increase'
    return str, index, prob

RawData1 = pd.read_excel('data.xlsx', sheetname='est')
Data1 = np.array(RawData1.iloc[0:30, 1:4])
allX1 = np.array(Data1[0:len(Data1), 0:2])
allY1 = np.array(Data1[0:len(Data1), 2])

RawData2 = pd.read_excel('data2.xlsx', sheetname='est')
Data2 = np.array(RawData2.iloc[0:30, 1:15])
allX2 = np.array(Data2[0:len(Data2), 0:13])
allY2 = np.array(Data2[0:len(Data2), 13])

sol1 = 0
sol2 = 0
test_numfiles = len(allX1)
risk = RiskZone_Criterion(allX1)
est = Estimation_Index_updown(allX2)
for ind in range(test_numfiles):
    str1, index1, prob1 = Result1(risk[ind])
    str2, index2, prob2 = Result2(est[ind])

    if index1 == allY1[ind]:
        sol1 = sol1 + 1
    if index2 == allY2[ind]:
        sol2 = sol2 + 1

    print('Result', "%02d" % (ind + 1),
          ' | result : ',
          index1, '\t',
          allY1[ind], '\t,',
          prob1, '\t',
          index2, '\t',
          allY2[ind], '\t,',
          prob2)

print(sol1/30*100, '\t', sol2/30*100)
