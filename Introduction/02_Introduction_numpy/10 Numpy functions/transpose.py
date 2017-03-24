import numpy as np
import tensorflow as tf

# np.transpose for (2, 2) works as expected
a = np.array([[2, 2], [4, 4]])
b = np.transpose(a)

print type(a), a.shape
print a
print type(b), b.shape
print b

print '================================================='

# np.transpose for (2, ) does not
c = np.array([2, 2])
d = np.transpose(c)

print type(c), c.shape
print c
print type(d), d.shape
print d

# tf.transpose for (2, ) does not work either
e = tf.constant([2, 2])
f = tf.transpose(e)

with tf.Session() as sess:
    e_run, f_run = sess.run([e, f])

    print type(e_run), e_run.shape
    print e_run
    print type(f_run), f_run.shape
    print f_run

print '================================================='

# np.transpose for (1, 2) works as expected
c2 = np.array([[2, 2]])
d2 = np.transpose(c2)

print type(c2), c2.shape
print c2
print type(d2), d2.shape
print d2

# tf.transpose for (1, 2) works as expected
e2 = tf.constant([[2, 2]])
f2 = tf.transpose(e2)

with tf.Session() as sess:
    e2_run, f2_run = sess.run([e2, f2])

    print type(e2_run), e2_run.shape
    print e2_run
    print type(f2_run), f2_run.shape
    print f2_run