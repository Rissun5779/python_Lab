import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 5, 0.01)
f = 1
pi = np.pi

y = np.sin(2*pi*f*t)

fig, ax = plt.subplots()
ax.plot(t, y)
plt.show()