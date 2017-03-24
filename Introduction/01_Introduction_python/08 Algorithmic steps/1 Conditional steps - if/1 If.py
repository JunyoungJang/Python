# -*- coding: utf8 -*-


'''
print 'Comparison operations: ==, !=, >, >=, <, <= ---'
x = 5
if x == 5:
    print 'Equals 5'
    print 1+1
if x != 6:
    print 'Not equals 5'
if x > 4:
    print 'Greater than 4'
if x >= 4: print 'Greater than or Equals 4'
if x < 6: print 'Less than 6'
if x <= 5: print 'Less than or Equals 5'
'''
'''
'''
#'''
#'''
#'''
print 'Warning: Turn Off Tabs (Edit > Convert Indents > To Spaces)'
x = 5
if x == 5:
    print 'Equals 5' # Use tab
    print 1+1 # Use 4 spaces
#'''
'''
'''
#'''
#'''
'''
print 'If something such as string, list, tuple, dictionary is empty or 0, the value is false. Otherwise it is true. ---'
if 0: # 0 - False
# if 1:
    print ("True")
else :
    print ("False")
if False: # False - False
# if True:
    print ("True")
else :
    print ("False")
if "": # empty string - False
# if 'python':
    print ("True")
else :
    print ("False")
if []: # empty list - False
# if [1,2,3]:
    print ("True")
else :
    print ("False")
if (): # empty tuple - False
# if (1,2,3):
    print ("True")
else :
    print ("False")
if {}: # empty dictionary - False
# if {'cat': 'cute', 'dog': 'furry'}:
    print ("True")
else :
    print ("False")
if None: # None - False
    print ("True")
else :
    print ("False")
'''
'''
'''
#'''
#'''
'''
print 'One-way if ---'
money = 1
if money:
    print 'Go there by taxi!'
'''
'''
'''
#'''
#'''
'''
print 'Two-wayway if - example 1 ---'
money = 1
if money:
    print 'Go there by taxi!'
else:
    print 'Go there on foot!'
'''
'''
'''
#'''
#'''
'''
print 'Two-wayway if - example 2 ---'
money = 2
if money >= 3:
    print 'Go there by taxi!'
else:
    print 'Go there on foot!'
'''
'''
'''
#'''
#'''
'''
print 'Multi-way if - example 1 ---'
money = 2
credit_card = 1
if money >= 3:
    print 'Go there by taxi!'
elif credit_card:
    print 'Go there by taxi!'
else:
    print 'Go there on foot!'
'''
'''
'''
#'''
#'''
'''
print 'Multi-way if - example 2 ---'
# No else at the end is OK
money = 2
credit_card = 1
if money >= 3:
    print 'Go there by taxi!'
elif credit_card:
    print 'Go there by taxi!'
# else:
    # print 'Go there on foot!'
'''
'''
'''
#'''
#'''
'''
print 'Multi-way if - puzzle 1 ---'
# Which lines are never executed for any choice of x value
x = 20
if x < 2:
    print 'Below 2'
elif x >= 2:
    print 'Two or more'
else:
    print 'Something else'
'''
'''
'''
#'''
#'''
'''
print 'Multi-way if - puzzle 2 ---'
# Which lines are never executed for any choice of x value
x = 20
if x < 2:
    print 'Below 2'
elif x < 20:
    print 'Below 20'
elif x < 10:
    print 'Below 10'
else:
    print 'Something else'
'''
'''
'''
#'''
#'''
'''
print 'if in ---'
# string with a logical operator in
fruit = 'banana'
print 'n' in fruit   # True
print 'm' in fruit   # False
print 'nan' in fruit # True

if 'a' in fruit:
    print 'Found it!'

if 4 in [1, 2, 3, 4]:
    print '4 is in the list'

pocket = ['paper', 'money', 'cellphone']
if 'money' in pocket:
    pass
else:
    print 'We accept a credit card.'

gases = ['He', 'Ne', 'Ar', 'Kr']
print 'He' in gases     # True  / False
print 'Xe' in gases     # True  / False
if 'Pu' in gases:
    print 'Pu , plutonium is not a gas!'
else:
    print 'The universe is well orderd.'
'''
'''
'''
#'''
#'''
'''
print 'if not ---'
coffee = 10
money = 0
if not coffee:
    print '커피가 다 떨어졌습니다. 판매를 중지합니다. 죄송합니다.'
if not money:
    print '커피는 한잔에 1000원 입니다. 커피를 드시고 싶으면 돈을 주세요. 죄송합니다'
'''
'''
'''
#'''
#'''
#'''
print 'if is, if is not ---'
largest_number_so_far = None
smallest_number_so_far = True
if largest_number_so_far is None:
    print 'We don\'t start the project of finding the largest number'
if smallest_number_so_far is not False:
    print 'We start the project of finding the smallest number'
#'''
'''
print 'inline if ---'
a = 2
print('Positive' if a >= 0 else 'Negative')
'''
'''
'''
#'''
#'''
#'''

