import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

f = open("losses.txt", 'r')
trainlosses = []
for line in f.readlines():
    line = line.strip()
    if line == "Train losses":
        continue
    elif line == "Val losses":
        continue
    trainlosses.append(float(line))
x = np.arange(0, len(trainlosses))
y = np.array(trainlosses)
print(y.max())
spl = make_interp_spline(x,y)

x = np.linspace(x.min(), x.max(), 10)
y_smooth = spl(x)

plt.plot(x, y_smooth)
plt.show()
