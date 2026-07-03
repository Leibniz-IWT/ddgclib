# neatplot
Style guideline for plotting data with Python using stylesheets.

# Tutorial

[For detailed explanations please see the accompanying tutorial notebook.](https://github.com/Leibniz-IWT/neatplot/blob/main/stylesheets_intro.ipynb) A minimum working example:

```python
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
```
output:

<img src="https://github.com/Leibniz-IWT/neatplot/blob/main/Figure_1.png" width="500">

