from matplotlib import pyplot as plt
import numpy as np

# skrive matten med font

def f_1(t, a=1, b=1, c=1, d=1):
    return np.sin(a*t) * np.cos(c*t), np.cos(b*t) * np.sin(d*t) 

def f_2(t, a=1, b=1, c=1, d=1):
    return np.sin(a*t), np.cos(b*t+c)

def f_3(t, k, l):
    return (1 - k)*np.cos(t) + l * k * np.cos((1-k)/k * t), (1 - k)*np.sin(t) - l * k * np.sin((1-k)/k * t)

def f_4(t, a, b, c, d, e, f, g, h):
    return a * np.cos(b*t) + c * np.cos(d * t), e * np.sin(f*t) - g * np.sin(h * t)

t = np.linspace(0, 4*np.pi, 1000)

fig, axs = plt.subplots(6, 6)

a_test = (0, 0.2, 0.5, 0.8, 1.0, 1.2)
b_test = (-0.2, 0.1, 0.6, 0.9, 1.1)
for i, a in enumerate(a_test):
    for j, b in enumerate(b_test):
        # x, y = f_1(t, a=a, b=b, c=5, d=2)
        # x, y = f_2(t, a=a, b=3, c=b, d=1)
        x, y = f_3(t, l=a, k=b)
        # x, y = f_4(t, a=0.4, b=2, c=0.3, d=1, e=0.2, f=3, g=0.2, h=2)
        # x, y = f_4(t, a=1.0, b=a, c=1.0, d=5, e=1.0, f=b, g=1.0, h=2)

        axs[i,j].plot(x, y)

plt.show()
