import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = a + b

print(a) # an operation
print(b) # an operation
print(c) # an operation

with tf.Session() as sess:
    print(c) # an operation
    print sess.run(c) # an operation executed
