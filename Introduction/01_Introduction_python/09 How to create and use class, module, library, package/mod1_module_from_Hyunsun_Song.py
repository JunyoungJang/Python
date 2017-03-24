# -*- coding: utf8 -*-
# mod1.py

def sum(a, b):
    return a + b

def safe_sum(a, b):
    if type(a) != type(b):
        print("더할수 있는 것이 아닙니다.")
        return
    else:
        result = sum(a, b)
    return result

# 직접 이 파일을 실행시켰을때는 name ==" main "이 참이되어 if문 다음문장들이 수행되고,
# 반대로 대화형 인터프리터나 다른 파일에서 이 모듈을 불러서 쓸때는 name == " main "이 거짓이 되어 if문 아래문장들이 수행되지 않도록 한다는 뜻이다.
if __name__ == "__main__":
    print(safe_sum('a', 1))
    print(safe_sum(1, 4))
    print(sum(10, 10.4))

'''
==================================================
Exercise
==================================================
note : module should exist same file or library

1.
import mod1
print(mod1.safe_sum(3,4))
print(mod1.safe_sum(3,'a'))
print(mod1.sum(3,4))

2.
from mod1 import sum
sum(3,4)

3.
from mod1 import sum, safe_sum
from mod1 import *
'''
