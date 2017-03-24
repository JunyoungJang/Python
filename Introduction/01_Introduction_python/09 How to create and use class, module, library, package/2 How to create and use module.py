# -*- coding: utf8 -*-

import mod1_module_from_Hyunsun_Song
print mod1_module_from_Hyunsun_Song.sum(3,4)
print mod1_module_from_Hyunsun_Song.safe_sum(3, 4)
print mod1_module_from_Hyunsun_Song.safe_sum(1, 'a') # return 이 없거나 단독으로 쓰인 return에 의해서 'None'이 추가적으로 나다.
print mod1_module_from_Hyunsun_Song.safe_sum(1, [2, 3])

from mod1_module_from_Hyunsun_Song import safe_sum
print safe_sum(3, 4)
print safe_sum(1, 'a') # return 이 없거나 단독으로 쓰인 return에 의해서 'None'이 추가적으로 나다.
print safe_sum(1, [2, 3])

from mod1_module_from_Hyunsun_Song import sum, safe_sum
print sum(3,4)
print safe_sum(3, 4)
print safe_sum(1, 'a') # return 이 없거나 단독으로 쓰인 return에 의해서 'None'이 추가적으로 나다.
print safe_sum(1, [2, 3])

from mod1_module_from_Hyunsun_Song import *
print sum(3,4)
print safe_sum(3, 4)
print safe_sum(1, 'a') # return 이 없거나 단독으로 쓰인 return에 의해서 'None'이 추가적으로 나다.
print safe_sum(1, [2, 3])

import mod2_module_from_Hyunsun_Song as mod2
print mod2.PI
a = mod2.Math()
print a.solv(2)
print mod2.sum(3,4)
print mod2.sum(mod2.PI, 4.4)


'''
==================================================
Exercise
==================================================

1.Add module to same file from another file

import sys

sys.path
sys.path.append

2.Applying change elements

import imp
imp.reload(mod2)
print(mod2.PI)
'''