import tensorflow as tf

a = tf.Variable([2.], dtype=tf.float32, name='Variable_a')
b = tf.Variable([3.], dtype=tf.float32, name='Variable_b')
c = tf.Variable([4.], dtype=tf.float32, name='Variable_c')

result = a + b * c

with tf.Session() as Simple_Operation:
    
    # to run an expression containing tf.Variable, we need to either initialize or restore
    tf.global_variables_initializer().run()
    output = Simple_Operation.run(result)
    print output
