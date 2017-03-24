# -*- coding: utf8 -*-


'''
print 'try except - example 1 ---'
a = 'Hello Bob'
try:
    b = int(a)
except:
    print 'something wrong with this step'
    b = -1
print 'First', b
a = '123'
try:
    b = int(a)
except:
    print 'something wrong with this step'
    b = -1
print 'Second', b
'''
'''
'''
#'''
#'''
'''
print 'try except - example 2 ---'
a = 'Bob'
try:
    print 'Hello'
    b = int(a)
    print b
except:
    b = -1
print 'Done', b
'''
'''
'''
#'''
#'''
'''
print 'try except - example 3 ---'
astr = raw_input('Enter your birth year : ')
try:
    istr = int(astr)
except:
    istr = -1

if istr > 0:
    print 'Nice work'
else:
    print 'Not a number'
'''
'''
'''
#'''
#'''
'''
print 'try except - example 4 ---'
# From Hours and Rate, compute Pay
Hours = raw_input('Enter Hours : ')
try:
    float_Hours = float(Hours)
    Error_code_1 = False
except:
    Error_code_1 = True

Rate = raw_input('Enter Rate : ')
try:
    float_Rate = float(Rate)
    Error_code_2 = False
except:
    Error_code_2 = True

if Error_code_1:
    print 'Error_code_1 active : Hours is not entered as a number'
elif Error_code_2:
    print 'Error_code_2 active : Rate is not entered as a number'
else:
    Pay = float_Hours * float_Rate
    print 'Hours :', Hours
    print 'Rate :', Rate
    print 'Pay :', Pay
'''
'''
'''
#'''
#'''
#'''
print 'try except - example 5 ---'
# From Hours and Rate, compute Pay. Over time rate (above 40 hours) is 1.5 times rate.
Hours = raw_input('Enter Hours : ')
try:
    float_Hours = float(Hours)
    Error_code_1 = False
except:
    Error_code_1 = True

Rate = raw_input('Enter Rate : ')
try:
    float_Rate = float(Rate)
    Error_code_2 = False
except:
    Error_code_2 = True

if Error_code_1:
    print 'Error_code_1 active : Hours is not entered as a number'
elif Error_code_2:
    print 'Error_code_2 active : Rate is not entered as a number'
else:
    Pay = min(float_Hours, 40) * float_Rate + max(float_Hours-40, 0) * (1.5*float_Rate)
    print 'Hours :', Hours
    print 'Rate :', Rate
    print 'Overtime Rate (over 40 hours) :', 1.5*float_Rate
    print 'Pay :', Pay
#'''
