# Programme 11.3

import numpy as np
mon_tableau = np.array([[10, 20], [30, 40]])
mon_autre_tableau = np.copy(mon_tableau)

mon_tableau[0][1] = 200
print(mon_tableau)
print(mon_autre_tableau)

