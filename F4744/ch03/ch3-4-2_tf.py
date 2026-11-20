import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
print("a=")
print(a.numpy())

s = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
print("s=")
print(s.numpy())

b = tf.add(a, s)
print("a + s=")
print(b.numpy())

b = tf.subtract(a, s)
print("a - s=")
print(b.numpy())

b = tf.multiply(a, s)
print("a * s=")
print(b.numpy())

b = tf.divide(a, s)
print("a / s=")
print(b.numpy())
