import tensorflow as tf

x = tf.constant([[1.2, 5.5, 8.7, 8.5],
                 [2.2, 4.3, 6.5, 9.5],
                 [6.2, 7.3, 1.5, 3.5]])
print(x)
print(tf.rank(x).numpy())
print(x.shape)
