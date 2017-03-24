import tensorflow as tf

LOG_DIR = "./temp/logfile"
SAVE_PATH = LOG_DIR + "/savefile.ckpt" # Where to save. This line is changed --------------------------------------------------------------------

x_2 = tf.Variable(0, dtype=tf.float32, name='x_2')
y_2 = tf.Variable(0, dtype=tf.float32, name='y_2')
z_2 = tf.Variable(0, dtype=tf.float32, name='z_2')

with tf.Session() as Recall_Linear_regression: # This line is changed ------------------------------------------------------------
    tf.global_variables_initializer().run()
    saver = tf.train.Saver( {'x': x_2, 'y': y_2, 'z': z_2} ) # This line is changed ------------------------------------------------
    saver.restore(Recall_Linear_regression, SAVE_PATH)  # This line is changed ---------------------------------------------------
    print(x_2, y_2, z_2)