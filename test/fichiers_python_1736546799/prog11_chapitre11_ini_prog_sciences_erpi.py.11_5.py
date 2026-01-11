# Programme 11.5

import matplotlib.pyplot as plt
import numpy as np

coord_x = np.linspace(0, 2, 200)
coord_y1 = coord_x
coord_y2 = coord_x**2
coord_y3 = np.sqrt(coord_x)
plt.figure(figsize=(3, 3))
plt.plot(coord_x, coord_y1, "r")
plt.plot(coord_x, coord_y2, "g")
plt.plot(coord_x, coord_y3, "b")
plt.grid()
plt.show()

