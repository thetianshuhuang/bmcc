
from matplotlib import pyplot as plt
import math
import numpy as np
import bmcc

v_n = bmcc.MFM(
    gamma=1, prior=lambda k: k * math.log(3 / 4)
).get_args(np.zeros(200))["V_n"]


res = []
for i in range(len(v_n) - 1):
    res.append(v_n[i + 1] - v_n[i])

plt.plot(res)

plt.show()
