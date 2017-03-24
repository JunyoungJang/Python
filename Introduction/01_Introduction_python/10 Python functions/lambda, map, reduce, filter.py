#-*- coding: utf-8 -*-
from functools import reduce

# (lambda 인자: 표현식) (구체적인 인자값) - 함수 이름도 없이 함수를 딱 한 줄만으로 만들게 해주는 훌륭한 녀석입니다.
print (lambda x, y: x + y) (10, 20)
k = [lambda a, b: a + b, lambda a, b: a * b]
print k
print k[0](3, 4)
print k[1](3, 4)

# map(함수, 리스트) - 리스트로부터 원소를 하나씩 꺼내서 함수를 적용시킨 다음, 그 결과를 새로운 리스트에 담아준답니다.
print map(lambda x: x ** 2, range(5)) # [0, 1, 4, 9, 16]

# reduce(함수, 순서형 자료) - 순서형 자료(문자열, 리스트, 튜플)의 원소들을 누적적으로(cummulative in MATLAB) 함수에 적용시킨답니다.
print reduce(lambda x, y: x + y, [0, 1, 2, 3, 4]) # 10
print reduce(lambda x, y: x + y, 'abcde') # abcde
print reduce(lambda x, y: y + x, 'abcde') # edcba

# filter(조건, 리스트) - 리스트로부터 원소를 하나씩 꺼내서 조건이 만족하는지 확인하고, 만족하 그 결과를 새로운 리스트에 담아준답니다.
print filter(lambda x: x < 5, range(10)) # [0, 1, 2, 3, 4]
print filter(lambda x: x % 2, range(10)) # [1, 3, 5, 7, 9]
print filter(lambda x: (x % 2) == 0, range(10)) # [0, 2, 4, 6, 8]