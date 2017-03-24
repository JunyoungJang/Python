import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.DataFrame([[40., 170., 70.], [20., 180., 60.]], columns=['Age','Height','Weight'], index=['Lee','Kim'])
df = np.array(df)

mean_1 = tf.reduce_mean(df, 0)
mean_2 = tf.reduce_mean(tf.reduce_mean(df, 0))

with tf.Session() as sess:

    print sess.run(mean_1)
    print sess.run(mean_2)
