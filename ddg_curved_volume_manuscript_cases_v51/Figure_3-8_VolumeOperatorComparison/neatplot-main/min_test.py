import matplotlib.pyplot as plt
import numpy as np
plt.style.use('standard.mplstyle')

# make data
x = np.linspace(0, 10, 100)
y = 4 + 2 * np.sin(2 * x)
# plot
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()