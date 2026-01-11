# Programme 11.11

import numpy as np

notes = [74, 86, 87, 99, 94, 81, 91, 98]
moyenne = np.mean(notes)
ecart_type = np.std(notes)
print("Moyenne de", round(moyenne, 1),
      "avec un écart type de", round(ecart_type, 1))

