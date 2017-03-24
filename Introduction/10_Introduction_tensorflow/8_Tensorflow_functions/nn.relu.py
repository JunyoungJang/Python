import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-5.0, 5.0, 100)
y = tf.nn.relu(x)

with tf.Session() as sess:
    x_eval, y_eval = sess.run([x, y])
    plt.plot(x_eval, y_eval)
    plt.axis([-5, 5, -1, 6])
    plt.show()

