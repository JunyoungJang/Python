# -*- coding: utf8 -*-
import numpy as np

'''
print 'iteration over list - example 1 ---'
email_list = ['me@gmail.com', 'you@hotmail.com', 'them@gmail.com']
for email in email_list:
    print(email)

for email in email_list:
    if 'gmail' in email: # This line is changed ----------------------------------------------------------------------------------
        print(email) # This line is changed --------------------------------------------------------------------------------------
'''
'''
#'''
#'''
'''
'''
print 'iteration over list - example 2 ---'
# for loop를 이용 1부터 100까지 자연수의 합을 구하라.
sum = 0
for i in range(101): # range(101) = [0, 1, 2, ..., 100]
    sum += i # sum = sum + i
print '1부터 100까지 자연수의 합 : ', sum
print np.sum(range(101))
'''
'''
'''
#'''
#'''
'''
print 'iteration over list - example 3 ---'
# 0 1 2 3 4 5 6 7 8 9 한 줄에 출력하기
for i in range(10):
    print i,
'''
'''
'''
#'''
#'''
'''
print 'iteration over list - example 4 ---'
# for loop를 이용 구구단 표를 프린트하기
for i in range(1,10): # range(1,10) = [1, 2, 3, …, 9]
    for j in range(1,10):
        print '%d * %d = %d' %(i, j, i*j)
'''
'''
'''
#'''
#'''
'''
print 'iteration over list - example 5 ---'
# use range to construct lists of numbers
gases = ['He', 'Ne', 'Ar', 'Kr']
print len(gases)   # 4
print range(len(gases)) # [0, 1, 2, 3]
for i in range(len(gases)): # a very common idiom in Python
    print i, gases[i]
'''
'''
'''
#'''
#'''
'''
print 'iteration over list - example 6 ---'
for i in [5,4,3,2,1]:
    print i

print 'Blastoff!'
'''
'''
'''
#'''
#'''
'''
print 'iteration over list - example 7 ---'
friends = ['Bob', 'Glenn', 'Tom']
for friend in friends:
    print 'Happy New Year', friend

print 'Done!'
'''
'''
'''
#'''
#'''
'''
print 'iteration over list - example 8 ---'
# 총 5명의 학생 시험점수가 90, 25, 67, 45, 80인데 60점이 넘으면 합격이다. 각각의 학생이 합격인지 불합격인지 프린트하라.
scores=[90,25,67,45,80]
number=0
for score in scores:
    number=number+1
    if score>60:
        print '%d번 학생은 합격입니다. ' %number
    else:
        print '%d번 학생은 불합격 입니다.' %number
'''
'''
'''
#'''
#'''
'''
print 'iteration over list - example 9 ---'
# 총 5명의 학생 시험점수가 90, 25, 67, 45, 80인데 60점이 넘으면 합격이고 그렇지 않으면 불합격이다. 합격한 학생들에게 축하메세지를 보낸다.
scores=[90,25,67,45,80]
number=0
for score in scores:
    number=number+1
    if score>60:
        print '%d번 학생은 합격입니다. 축하합니다.' %number
'''
'''
'''
#'''
#'''
'''
print 'iteration over list - example 10 ---'
person_list = ['me', 'you', 'them']
email_server_list = ['gmail', 'hotmail', 'gmail']
print zip(person_list, email_server_list)

for person, email_server in zip(person_list, email_server_list):
    print person, email_server
    print person +'@' + email_server + '.com'
'''
'''
'''
#'''
#'''
'''
print 'loop idiom - dumb initialization - example 1 ---'
# find largest number
x_list = [9, 41, 12, 3, 74, 15]
largest_number_so_far = x_list[0]
for x in x_list:
    if largest_number_so_far is None:
        largest_number_so_far = x
    if x > largest_number_so_far:
        largest_number_so_far = x

print 'Largest number', largest_number_so_far
'''
'''
'''
#'''
#'''
'''
print 'loop idiom - smart initialization - example 2 ---'
# find largest number and place
x_list = [9, 41, 12, 3, 74, 15]
count = 0
total = 0
largest_number_so_far = None
largest_number_so_far_index = 0
for i, x in enumerate(x_list):
    if largest_number_so_far is None:
        largest_number_so_far = x
    count = count + 1
    total = total + x
    print count, total, x
    if x > largest_number_so_far:
        largest_number_so_far = x
        largest_number_so_far_index = i

print 'Total number of x in the list : ', count
print 'Total sum of all x in the list : ', total
print 'Largest number : ', largest_number_so_far
print 'Largest number index : ', largest_number_so_far_index
'''
'''
'''
#'''
#'''
'''
print 'loop idiom - example 3 ---'
# filter and collect all numbers x >= 20
x_list = [9, 41, 12, 3, 74, 15]
x_filtered = []
for x in x_list:
    if x >= 20:
        x_filtered.append(x)

print 'Filted x in the list : ', x_filtered
'''
'''
'''
#'''
#'''
'''
print 'loop idiom - checker initialization - example 4 ---'
# decide whether there is a number 3 in the list
x_list = [9, 41, 12, 3, 74, 15]
found_checker = False
print 'Before : ', found_checker
for x in x_list:
    if x == 3:
        found_checker = True
        print found_checker, x
        break
    print found_checker, x

print 'After : ', found_checker
'''



