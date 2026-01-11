# Programme 2.5

import numpy as np

# spécifier les longueurs de l'hypoténuse
# et d'un autre des côtés d'un triangle rectangle
hypotenuse = 5
cote = 4        # doit être < hypotenuse

angle_oppose_rad = np.arcsin(cote/hypotenuse)
angle_adjacent_rad = np.arccos(cote/hypotenuse)
angle_oppose_deg = np.degrees(angle_oppose_rad)
angle_adjacent_deg = np.degrees(angle_adjacent_rad)
print("Angles autres que l'angle droit: ",
      round(angle_oppose_deg, 1), "° et ",
      round(angle_adjacent_deg, 1), "°", sep="")

