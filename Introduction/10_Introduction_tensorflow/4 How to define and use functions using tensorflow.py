#-*- coding: utf-8 -*-
# 주어진 숫자의 구구단 표 만들기

import tensorflow as tf

def table99(integer_given):
    left = tf.placeholder(tf.int32)
    right = tf.placeholder(tf.int32)
    left_right_multiplication = tf.mul(left, right)

    with tf.Session() as sess:
        for i in range(1, 10):
            multiplication_result = sess.run(left_right_multiplication, feed_dict={left: integer_given, right: i})
            print('%d x %d = %2d' %(integer_given, i, multiplication_result))

table99(7)

# Exercise
# Redefine the above function using tf.constant instead of tf.placeholder and run.

# Exercise
# Redefine the above function using tf.Variable instead of tf.placeholder and run.
