import tensorflow as tf

x = tf.constant([1.2, 5.5, 8.7, 10.5])
print(x)
print(tf.rank(x).numpy())
