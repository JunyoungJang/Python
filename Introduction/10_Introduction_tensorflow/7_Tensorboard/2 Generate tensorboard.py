import tensorflow as tf

LOG_DIR = './'

a = tf.constant([2.], dtype=tf.float32, name='Constant_a')
b = tf.constant([3.], dtype=tf.float32, name='Constant_b')
c = tf.constant([4.], dtype=tf.float32, name='Constant_c')

result = a + b * c

with tf.Session() as Simple_Operation:
    writer = tf.train.SummaryWriter('./', Simple_Operation.graph)
    output = Simple_Operation.run(result)
    print output

# cd Dropbox/Tensorflow/12\ Tensorboard/temp/1\ Simple operation using tf.constant as a tensorflow logs/
# tensorboard --logdir=./
# http://localhost:6006/
