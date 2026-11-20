import tensorflow as tf

x = tf.constant(10.5)
print(x)
print(tf.rank(x).numpy())
