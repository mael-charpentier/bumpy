# Programme 2.4

import numpy as np

# spécifier la longueur de l'hypoténuse et la valeur
# d'un des angles non-droits d'un triangle rectangle (en degrés)
hypotenuse = 10
angle_deg = 30        # doit être < 90°

angle_rad = angle_deg * np.pi/180
cote_oppose = hypotenuse * np.sin(angle_rad)
cote_adjacent = hypotenuse * np.cos(angle_rad)
print("Longueur du côté opposé: ", format(cote_oppose, ".3g"))
print("Longueur du côté adjacent: ", format(cote_adjacent, ".3g"))

