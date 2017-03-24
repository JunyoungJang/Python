# -*- coding: utf8 -*-
# mod2.py

PI = 3.141592 # 클래스나 변수등을 포함한 모듈 - 변수

class Math: # 클래스나 변수등을 포함한 모듈 - 클라스
    def solv(self, r):
        return PI * (r ** 2)

def sum(a, b):
    return a + b

if __name__ == "__main__":
    print(PI)
    a = Math()
    print(a.solv(2))
    print(sum(PI, 4.4))

'''
==================================================
Exercise
==================================================
note : __name__=="__main__" means
When you directly execute not using 'import' then 'print' command execute

1.
import mod2
a=mod2.Math()
print(a.solv(2))

2.
print(mod2.sum(mod2.PI, 4.4))
'''

