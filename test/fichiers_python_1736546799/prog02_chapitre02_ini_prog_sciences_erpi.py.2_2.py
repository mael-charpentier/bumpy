# Programme 2.2

import numpy as np

# coefficients de l'équation ax² + bx + c = 0 (a ≠ 0)
a = 1
b = 3
c = -10
discriminant = b**2 - 4*a*c
x_1 = (-b - np.sqrt(discriminant))/(2*a)
x_2 = (-b + np.sqrt(discriminant))/(2*a)
print("Solutions: x_1 =", x_1, "et x_2 =", x_2)

