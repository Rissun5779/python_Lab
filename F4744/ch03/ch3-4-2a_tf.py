import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
print("a=")
print(a.numpy())

s = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
print("s=")
print(s.numpy())

b = tf.linalg.matmul(a, s)
print("tf.linalg.matmul(a, s)=")
print(b.numpy())
