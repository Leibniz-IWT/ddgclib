import numpy as np
import matplotlib.pyplot as plt

N0, D = 1.0, 1e-9
t_values = [1, 10, 100]
x = np.linspace(-5e-4, 5e-4, 500)

def n_xt(x, t):
    return N0 / np.sqrt(4*np.pi*D*t) * np.exp(-x**2/(4*D*t))

plt.figure(figsize=(7,5))
for t in t_values:
    plt.plot(x*1e3, n_xt(x,t), label=f't = {t}s')
plt.title(r"Diffusion Solution: $n(x,t)=\frac{N_0}{\sqrt{4\pi D t}} e^{-x^2/(4Dt)}$")
plt.xlabel("x (mm)")
plt.ylabel("n(x,t)")
plt.legend(); plt.grid(True)
plt.show()
