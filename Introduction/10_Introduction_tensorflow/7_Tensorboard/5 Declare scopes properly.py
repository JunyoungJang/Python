import tensorflow as tf

LOG_DIR = "./temp/1 Simple operation using tf.constant as a tensorflow logs"

a = tf.constant([2.], dtype=tf.float32, name='Constant_a') # This line is changed ------------------------------------------------
b = tf.constant([3.], dtype=tf.float32, name='Constant_b') # This line is changed ------------------------------------------------
c = tf.constant([4.], dtype=tf.float32, name='Constant_c') # This line is changed ------------------------------------------------

with tf.name_scope('Input_layer_1') as Input_layer_1: # This line is changed -----------------------------------------------------
    result = a + b * c

d = tf.constant([2.1], dtype=tf.float32, name='Constant_d') # This line is changed -----------------------------------------------
e = tf.constant([3.1], dtype=tf.float32, name='Constant_e') # This line is changed -----------------------------------------------
f = tf.constant([4.1], dtype=tf.float32, name='Constant_f') # This line is changed -----------------------------------------------

with tf.name_scope('Input_layer_2') as Input_layer_2: # This line is changed -----------------------------------------------------
    result2 = d * e + f

with tf.Session() as Simple_Operation:
    writer = tf.summary.FileWriter(LOG_DIR, Simple_Operation.graph)
    output = Simple_Operation.run(result)
    print output

# cd Dropbox/Tensorflow/12\ Tensorboard/temp/1\ Simple operation using tf.constant as a tensorflow logs/
# tensorboard --logdir=./
# http://localhost:6006/

