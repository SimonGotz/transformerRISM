import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

f = open("../Results/Experiments(jan2025)/overfitTrain.txt", 'r')
trainlosses = []
vallosses = []
val = False
for line in f.readlines()[7:]:
    line = line.strip()
    if not val and line != "Val losses":
        trainlosses.append(float(line))
    if line == "Val losses":
        val = True
        continue
    if not val:
        continue
    vallosses.append(float(line))
x_train = np.arange(0, len(trainlosses))
y_train = np.array(trainlosses)
x_val = np.arange(0, len(vallosses))
y_val = np.array(vallosses)
#print(y.max())
spl_val = make_interp_spline(x_val,y_val)
spl_train = make_interp_spline(x_train,y_train)
x_val = np.linspace(x_val.min(), x_val.max(), 100)
x_train = np.linspace(x_train.min(), x_train.max(), 100)
y_val = spl_val(x_val)
y_train = spl_train(x_train)
plt.plot(x_train, y_train, label="Train")
plt.plot(x_val, y_val, label='Validation')
plt.legend()
plt.show()
