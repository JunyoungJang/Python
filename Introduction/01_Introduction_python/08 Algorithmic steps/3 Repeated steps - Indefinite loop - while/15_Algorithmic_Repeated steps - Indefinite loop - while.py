# -*- coding: utf8 -*-

password_user_typed = '1'
password_stored = 'python123'
while password_user_typed != password_stored:
    password_user_typed = raw_input('Enter password : ')
    if password_user_typed == password_stored:
        print('You are logged in.')
    else:
        print('Password is not correct. Please try again!')


'''
print 'iteration variable - example 1 ---'
# 열 번 찍어 안 넘어 가는 나무 없다.
TreeHit = 0
while TreeHit < 10:
    TreeHit += 1
    print "나무를 %d번 찍었습니다." %TreeHit
    if TreeHit == 10:
        print '나무 드디어 넘어갑니다.'
'''
'''
'''
#'''
#'''
'''
print 'iteration variable - example 2 ---'
# 10이하의 홀수를 출력하라.
a = 1
while a < 10:
    a += 2
'''
'''
'''
#'''
#'''
'''
print 'iteration variable - example 3 ---'
# 10이하의 양수를 반대 순서로 출력하라.
a = 9
while a > 0:
    print a
    a -= 1
'''
'''
'''
#'''
#'''
'''
print 'iteration variable - example 4 ---'
# print all prime numbers less than 1000 - wise version
n = 2
while n < 1000:
    # step 1 - figure out if n is prime
    is_prime = True
    m = 2
    while m**2 <= n: # m <= math.sqrt(n)
        if (n%m) == 0: # n is divisible by m
            is_prime = False
        m += 1
    # step 2 - print out n if n is prime
    if is_prime == True:
        print n,
    # step 3 - go to next number
    n += 1
'''
'''
'''
#'''
#'''
'''
print 'iteration variable - example 5 ---'
# 1원짜리 커피 장사하기
coffee = 10
money = 5
while money:
    if not coffee:
        print '커피가 다 떨어졌습니다. 판매를 중지합니다. 죄송합니다.'
        break
    if not money:
        print '커피는 공짜가 아닙니다. 커피를 드시고 싶으면 돈을 가져오세요. 죄송합니다.'
        break
    coffee -= 1
    money -= 1
    print '남은 커피의 양은 %d잔 입니다.' % coffee
    print '주문하신 커피와 거스름 돈 %d원 여기있습니다.' % money
    if not coffee:
        print '커피가 다 떨어졌습니다. 판매를 중지합니다. 죄송합니다.'
        break
'''
'''
'''
#'''
#'''
'''
print 'iteration variable - example 6 ---'
# 1000원짜리 커피 장사하기
coffee = 20
money = 10000
while money:
    if not coffee:
        print '커피가 다 떨어졌습니다. 판매를 중지합니다. 죄송합니다.'
        break
    if money < 1000:
        print '커피는 한잔에 1000원 입니다. 커피를 드시고 싶으면 돈을 더 가져오세요. 죄송합니다'
        break
    coffee -= 1
    money -= 1000
    print '남은 커피의 양은 %d잔 입니다.' % coffee
    print '주문하신 커피와 거스름 돈 %d원 여기있습니다.' % money
    if not coffee:
        print '커피가 다 떨어졌습니다. 판매를 중지합니다. 죄송합니다.'
        break
'''
'''
'''
#'''
#'''
'''
print 'iteration variable - example 7 - infinite loop ---'
# How to get out of the infinite while loop - Command + fn + f2
n = 5
while n > 0:
    print 'Lather'
    print 'Rinse'

print 'Dry off'
'''
'''
'''
#'''
#'''
'''
print 'iteration variable - example 8 - no loop ---'
n = 0
while n > 0:
    print 'Lather'
    print 'Rinse'

print 'Dry off'
'''
'''
'''
#'''
#'''
'''
print 'Going out of a loop with break ---'
while True:
    line = raw_input('>>>')
    print line
    if line == 'done':
        break
print 'Done!'
'''
'''
'''
#'''
#'''
'''
print 'Going top of a loop with continue ---'
while True:
    line = raw_input('>>>')
    print line
    if line[0] == '#':
        continue
    if line == 'done':
        break
print 'Done!'
'''
'''
'''
#'''
#'''
'''
print 'loop over all elements in a list ---'
gases = ['He', 'Ne', 'Ar', 'Kr']
i = 0
while i < len(gases):
    print gases[i]
    i += 1
'''