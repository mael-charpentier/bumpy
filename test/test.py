import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import bumpy as bp


do_print = False

bp_a = bp.array([[[111,112,113],[121,122,123]],
           [[211,212,213],[221,222,223]],
           [[311,312,313],[321,322,323]],
           [[411,412,413],[421,422,423]],
           [[511,512,513],[521,522,523]]])

np_a = np.array([[[111,112,113],[121,122,123]],
                 [[211,212,213],[221,222,223]],
                 [[311,312,313],[321,322,323]],
                 [[411,412,413],[421,422,423]],
                 [[511,512,513],[521,522,523]]])

assert len(bp_a) == len(np_a)
assert bp_a.shape == np_a.shape
assert bp_a.size == np_a.size
#assert bp_a.strides == np_a.strides
assert bp_a._offset == 0

assert bp_a[2] == np_a[2]
assert bp_a[2,1] == np_a[2,1]
assert bp_a[2,1,0] == np_a[2,1,0]

bp_a[0] = [[99999,99999,99999],[99999,99999,99999]]
bp_a[1,0] = [99999,99999,99999]
bp_a[1,1,0] = 99999

np_a[0] = [[99999,99999,99999],[99999,99999,99999]]
np_a[1,0] = [99999,99999,99999]
np_a[1,1,0] = 99999

assert bp_a == np_a

if do_print:
    for i in range(bp_a.shape[0]):
        print('bp_a['+repr(i)+'] =', bp_a[i])
        for j in range(bp_a.shape[1]):
            print('bp_a['+repr((i,j))+'] =', bp_a[(i,j)])
            for k in range(bp_a.shape[2]):
                index = (i,j,k)
                print('bp_a['+repr(index)+'] =', bp_a[index])

    print(bp.array([1]*1000))

bp_a = bp.array([[1, 2], [3, 4]])
bp_b = bp.array([[5, 6], [7, 8]])
np_a = np.array([[1, 2], [3, 4]])
np_b = np.array([[5, 6], [7, 8]])
assert bp_a + bp_b == np_a + np_b

bp_a = bp.array([[1, 2, 3], [4, 5, 6]])
bp_b = bp.array([10, 20, 30])
np_a = np.array([[1, 2, 3], [4, 5, 6]])
np_b = np.array([10, 20, 30])
assert bp_a + bp_b == np_a + np_b
assert bp_a + 10 == np_a + 10
assert 10 + bp_a == 10 + np_a
assert bp_a * 0 == np_a * 0
assert bp_a.reshape((3, 2)) == np_a.reshape((3, 2))
assert bp_a.transpose() == np_a.transpose()

assert bp_a == np_a
assert bp_a[1,:2] == np_a[1,:2]
assert bp_a[0:2,1:] == np_a[0:2,1:]
bp_a[1, :2] = [7, 8]
np_a[1, :2] = [7, 8]
assert bp_a == np_a
bp_a[0:2, 1:] = bp.array([[9, 10], [11, 12]])
np_a[0:2, 1:] = [[9, 10], [11, 12]]
assert bp_a == np_a

bp_a = bp.array([1, 2, 3])
bp_b = bp.array([4, 5, 6])
np_a = np.array([1, 2, 3])
np_b = np.array([4, 5, 6])
assert bp.dot(bp_a, bp_b) == np.dot(np_a, np_b)

bp_a = bp.array([[1, 2], [3, 4]])
bp_b = bp.array([[5, 6], [7, 8]])
np_a = np.array([[1, 2], [3, 4]])
np_b = np.array([[5, 6], [7, 8]])
#assert bp.linalg.matmul(bp_a, bp_b) == np.linalg.matmul(np_a, np_b)
assert bp.matmul(bp_a, bp_b) == np.matmul(np_a, np_b)

assert bp_a.sum() == np_a.sum()
assert bp_a.sum(axis=0) == np_a.sum(axis=0)
assert bp_a.sum(axis=1) == np_a.sum(axis=1)

assert bp_a.mean() == np_a.mean()
assert bp_a.mean(axis=0) == np_a.mean(axis=0)
assert bp_a.mean(axis=1) == np_a.mean(axis=1)

assert bp_a.max() == np_a.max()
assert bp_a.max(axis=0) == np_a.max(axis=0)
assert bp_a.max(axis=1) == np_a.max(axis=1)

assert bp_a.min() == np_a.min()
assert bp_a.min(axis=0) == np_a.min(axis=0)
assert bp_a.min(axis=1) == np_a.min(axis=1)

assert bp_a.flatten() == np_a.flatten()

assert bp_a.any() == np_a.any()
assert bp_a.all() == np_a.all()

bp_a = bp.array([[0, 0], [0, 0]])
np_a = np.array([[0, 0], [0, 0]])
assert bp_a.any() == np_a.any()
assert bp_a.all() == np_a.all()

bp_a = bp.array([[0, 0], [1, 0]])
np_a = np.array([[0, 0], [1, 0]])
assert bp_a.any() == np_a.any()
assert bp_a.all() == np_a.all()

bp_a = bp.array([[1,2,3],[4,5,6],[7,8,9]])
np_a = np.array([[1,2,3],[4,5,6],[7,8,9]])
assert bp_a[[0,2],[1,0]] == np_a[[0,2],[1,0]]
assert bp_a[[0],[1,0]] == np_a[[0],[1,0]]

assert bp_a[:,[0,2]] == np_a[:,[0,2]]
assert bp_a[[0,1,2],[2,1,0]] == np_a[[0,1,2],[2,1,0]]

bp_a[[0, 2], [1, 0]] = [20, 30]
np_a[[0, 2], [1, 0]] = [20, 30]
assert bp_a == np_a

bp_a = bp.array([[10, 20], [30, 40], [50, 60]])
np_a = np.array([[10, 20], [30, 40], [50, 60]])
assert bp_a[[0, 1],[1, 0]] == np_a[[0, 1],[1, 0]]

bp_a = bp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
np_a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
assert bp_a[[0, 1],[1, 0]] == np_a[[0, 1],[1, 0]]

bp_a = bp.array([[1, 2, 3], [4, 5, 6]])
bp_a[[0, 1], [2, 1]] = 99
np_a = np.array([[1, 2, 3], [4, 5, 6]])
np_a[[0, 1], [2, 1]] = 99
assert bp_a == np_a

bp_a = bp.array([[1, 2], [3, 4]])
bp_a[[0, 0], [1, 1]] = [5, 6]
np_a = np.array([[1, 2], [3, 4]])
np_a[[0, 0], [1, 1]] = [5, 6]
assert bp_a == np_a

bp_a = bp.array([[[1,2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]]])
bp_a[[0, 2], [1, 0]] = [[100, 101], [200, 201]]
np_a = np.array([[[1,2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]]])
np_a[[0, 2], [1, 0]] = [[100, 101], [200, 201]]
assert bp_a[0,1].tolist() == np_a[0,1].tolist()
assert bp_a[2,0].tolist() == np_a[2,0].tolist()

print("repeat func")

bp_a = bp.array([1, 2, 3])
np_a = np.array([1, 2, 3])
bp_r1 = bp.repeat(bp_a, 2)
np_r1 = np.repeat(np_a, 2)
assert bp_r1.shape == np_r1.shape
assert bp_r1 == np_r1

bp_b = bp.array([[1, 2], [3, 4]])
np_b = np.array([[1, 2], [3, 4]])

bp_r2 = bp.repeat(bp_b, 2, axis=0)
np_r2 = np.repeat(np_b, 2, axis=0)
assert bp_r2.shape == np_r2.shape
assert bp_r2 == np_r2

bp_r3 = bp.repeat(bp_b, 3, axis=1)
np_r3 = np.repeat(np_b, 3, axis=1)
assert bp_r3.shape == np_r3.shape
assert bp_r3 == np_r3

bp_r4 = bp.repeat(bp_b, 2)
np_r3 = np.repeat(np_b, 2)
assert bp_r4.shape == np_r3.shape
assert bp_r4 == np_r3

bp_x = bp.array([1, 2, 3])
bp_y = bp.array([4, 5])
np_x = np.array([1, 2, 3])
np_y = np.array([4, 5])

print("meshgrid func")

bp_X, bp_Y = bp.meshgrid(bp_x, bp_y, indexing='xy', sparse=False)
np_X, np_Y = np.meshgrid(np_x, np_y, indexing='xy', sparse=False)
assert bp_X.shape == np_X.shape
assert bp_Y.shape == np_Y.shape
assert bp_X == np_X
assert bp_Y == np_Y

bp_X2, bp_Y2 = bp.meshgrid(bp_x, bp_y, indexing='ij', sparse=False)
np_X2, np_Y2 = np.meshgrid(np_x, np_y, indexing='ij', sparse=False)
assert bp_X2.shape == np_X2.shape
assert bp_Y2.shape == np_Y2.shape
assert bp_X2 == np_X2
assert bp_Y2 == np_Y2

bp_Xs, bp_Ys = bp.meshgrid(bp_x, bp_y, indexing='xy', sparse=True)
np_Xs, np_Ys = np.meshgrid(np_x, np_y, indexing='xy', sparse=True)
assert bp_Xs.shape == np_Xs.shape
assert bp_Ys.shape == np_Ys.shape
assert bp_Xs == np_Xs
assert bp_Ys == np_Ys

bp_x3 = bp.array([1, 2])
bp_y3 = bp.array([3, 4])
bp_z3 = bp.array([5, 6])
np_x3 = np.array([1, 2])
np_y3 = np.array([3, 4])
np_z3 = np.array([5, 6])

bp_X3, bp_Y3, bp_Z3 = bp.meshgrid(bp_x3, bp_y3, bp_z3, indexing='ij', sparse=False)
np_X3, np_Y3, np_Z3 = np.meshgrid(np_x3, np_y3, np_z3, indexing='ij', sparse=False)
assert bp_X3.shape == np_X3.shape
assert bp_Y3.shape == np_Y3.shape
assert bp_Z3.shape == np_Z3.shape
assert bp_X3 == np_X3
assert bp_Y3 == np_Y3
assert bp_Z3 == np_Z3

bp_x = bp.array([1, 2, 3])
bp_y = bp.array([4, 5])
bp_z = bp.array([6, 7, 8, 9])
np_x = np.array([1, 2, 3])
np_y = np.array([4, 5])
np_z = np.array([6, 7, 8, 9])

bp_X, bp_Y, bp_Z = bp.meshgrid(bp_x, bp_y, bp_z, indexing='xy', sparse=False)
np_X, np_Y, np_Z = np.meshgrid(np_x, np_y, np_z, indexing='xy', sparse=False)
assert bp_X == np_X
assert bp_Y == np_Y
assert bp_Z == np_Z


print("arange func")

# Testing numpy.arange
assert bp.arange(5) == np.arange(5)  # [0 1 2 3 4]
assert bp.arange(1, 6) == np.arange(1, 6)  # [1 2 3 4 5]
assert bp.arange(0, 10, 2) == np.arange(0, 10, 2)  # [0 2 4 6 8]
assert bp.arange(5, 0, -1) == np.arange(5, 0, -1)  # [5 4 3 2 1]
assert bp.arange(0, 1, 0.2) == np.arange(0, 1, 0.2)  # [0.  0.2 0.4 0.6 0.8]

# Edge cases for arange
assert bp.arange(0) == np.arange(0)  # []
assert bp.arange(10, 10) == np.arange(10, 10)  # []
assert bp.arange(1, 5, -1) == np.arange(1, 5, -1)  # []
assert bp.arange(0, 1.1, 0.3) == np.arange(0, 1.1, 0.3)  # Floating point step

print("split/stack func")

assert bp.hsplit(bp.array([[1, 2, 3], [4, 5, 6]]), 3) == np.hsplit(np.array([[1, 2, 3], [4, 5, 6]]), 3)  # Horizontal split
assert bp.vsplit(bp.array([[1, 2, 3], [4, 5, 6]]), 2) == np.vsplit(np.array([[1, 2, 3], [4, 5, 6]]), 2)  # Vertical split
assert bp.vstack([[1, 2, 3], [4, 5, 6]]) == np.vstack([[1, 2, 3], [4, 5, 6]])  # Stack bp.arrays vertically
assert bp.hstack([[1, 2, 3], [4, 5, 6]]) == np.hstack([[1, 2, 3], [4, 5, 6]])  # Stack bp.arrays horizontally
assert bp.vstack([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) == np.vstack([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Stack bp.arrays vertically
assert bp.hstack([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) == np.hstack([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Stack bp.arrays horizontally
assert bp.column_stack([[1, 2, 3], [4, 5, 6]]) == np.column_stack([[1, 2, 3], [4, 5, 6]])  # Stack columns together


print("array func")

# Creating bp.arrays
np_arr = np.array([1, 2, 3, 4, 5])
bp_arr = bp.array([1, 2, 3, 4, 5])
assert bp.ones((2, 2)) == np.ones((2, 2))  # 2x2 bp.array of ones
assert bp.zeros((3, 3)) == np.zeros((3, 3))  # 3x3 bp.array of zeros
assert bp.zeros_like(bp_arr) == np.zeros_like(np_arr)  # Zeros with same shape as arr
assert bp.array([1, 2, 3]) == np.array([1, 2, 3])  # Create a numpy bp.array

# Shape and Transpose
assert bp.shape(bp_arr) == np.shape(np_arr)  # Shape of the bp.array

np_arr = np.array([1, 2, 3, 4, 5])
bp_arr = bp.array([1, 2, 3, 4, 5])
assert bp.where(bp_arr > 3, "Big", "Small") == np.where(np_arr > 3, "Big", "Small")  # Label elements
# Vector operations
assert bp.eye(3) == np.eye(3)  # Identity matrix of size 3

print("linspace")

# Testing numpy.linspace
assert bp.linspace(0, 10, 5) == np.linspace(0, 10, 5)  # [ 0.   2.5  5.   7.5 10. ]
assert bp.linspace(1, 2, 3) == np.linspace(1, 2, 3)  # [1.  1.5 2. ]
assert bp.linspace(10, 20, 6) == np.linspace(10, 20, 6)  # [10. 12. 14. 16. 18. 20.]
assert bp.linspace(5, -5, 11) == np.linspace(5, -5, 11)  # [  5.   4.  3.  2.  1.  0. -1. -2. -3. -4. -5.]

# Edge cases for linspace
assert bp.linspace(0, 1, 1) == np.linspace(0, 1, 1)  # [0.]
assert bp.linspace(3, 3, 4) == np.linspace(3, 3, 4)  # [3. 3. 3. 3.]
assert bp.linspace(0, 10, 0) == np.linspace(0, 10, 0)  # []

# Including endpoint=False
assert bp.linspace(0, 10, 5, endpoint=False) == np.linspace(0, 10, 5, endpoint=False)  # [ 0.  2.  4.  6.  8.]
assert bp.linspace(0, 1, 5, endpoint=False) == np.linspace(0, 1, 5, endpoint=False)  # [0.  0.2 0.4 0.6 0.8]

# Returning step size as well
np_linspace_values, np_step = np.linspace(0, 10, 5, retstep=True)
bp_linspace_values, bp_step = bp.linspace(0, 10, 5, retstep=True)
assert bp_linspace_values == np_linspace_values  # [ 0.   2.5  5.   7.5 10. ]
assert bp_step == np_step             # 2.5

print("math func")

assert bp.dot([1, 2], [3, 4]) == np.dot([1, 2], [3, 4])  # Dot product of vectors
assert bp.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == np.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])  # Matrix multiplication


# Rounding and limits
assert bp.floor(3.7) == np.floor(3.7)  # Floor function
assert bp.ceil(3.2) == np.ceil(3.2)  # Ceil function
assert bp.round(3.567, 2) == np.round(3.567, 2)  # Rounds to 2 decimal places

# Binary representation
assert bp.binary_repr(10) == np.binary_repr(10)  # Binary string of 10

# Basic arithmetic
assert bp.add(5, 3) == np.add(5, 3)  # Addition
assert bp.subtract(10, 4) == np.subtract(10, 4)  # Subtraction
assert bp.multiply(3, 4) == np.multiply(3, 4)  # Multiplication
assert bp.divide(10, 2) == np.divide(10, 2)  # Division
assert bp.prod([1, 2, 3, 4]) == np.prod([1, 2, 3, 4])  # Product of bp.array elements
assert bp.power(2, 3) == np.power(2, 3)  # 2^3 = 8
assert bp.sqrt(16) == np.sqrt(16)  # Square root of 16


assert bp.gradient([1, 2, 4, 7, 11]) == np.gradient([1, 2, 4, 7, 11])
assert bp.trapezoid([1, 2, 3], dx=1) == np.trapezoid([1, 2, 3], dx=1)
assert bp.gcd(12, 18) == np.gcd(12, 18)  # GCD of 12 and 18
assert bp.lcm(12, 18) == np.lcm(12, 18)  # LCM of 12 and 18


# Min, Max, Mean, Median, Standard deviation, Variance, Sum
np_arr = np.array([1, 2, 3, 4, 5])
bp_arr = bp.array([1, 2, 3, 4, 5])
assert bp.max(bp_arr) == np.max(np_arr)            # Maximum value
assert bp.median(bp_arr) == np.median(np_arr)         # Median
assert bp.min(bp_arr) == np.min(np_arr)            # Minimum value
assert bp.mean(bp_arr) == np.mean(np_arr)           # Mean
assert bp.std(bp_arr) == np.std(np_arr)            # Standard deviation
assert bp.var(bp_arr) == np.var(np_arr)            # Variance
assert bp.sum(bp_arr) == np.sum(np_arr)            # Sum of elements


assert bp.pi == np.pi

# Inverse hyperbolic functions
assert bp.arccosh(2) == np.arccosh(2)
assert bp.arcsinh(1) == np.arcsinh(1)
assert bp.arctanh(0.5) == np.arctanh(0.5)

# Degree/Radian conversion
assert bp.deg2rad(180) == np.deg2rad(180)        # Convert degrees to radians
assert bp.rad2deg(bp.pi) == np.rad2deg(np.pi)      # Convert radians to degrees
assert bp.deg2rad(-180) == np.deg2rad(-180)        # Convert degrees to radians
assert bp.rad2deg(-bp.pi) == np.rad2deg(-np.pi)      # Convert radians to degrees

assert bp.hypot(3, 4) == np.hypot(3, 4)         # Hypotenuse of a right triangle

# Trigonometric functions
assert bp.cos(bp.pi) == np.cos(np.pi)          # Cosine
assert bp.cos(bp_arr) == np.cos(np_arr)            # Cosine
assert bp.sin(bp.pi / 2) == np.sin(np.pi / 2)      # Sine
assert bp.tan(bp.pi / 4) == np.tan(np.pi / 4)      # Tangent
assert bp.sinh(1) == np.sinh(1)             # Hyperbolic sine
assert bp.cosh(1) == np.cosh(1)             # Hyperbolic cosine
assert bp.tanh(1) == np.tanh(1)             # Hyperbolic tangent
assert bp.arccos(0.5) == np.arccos(0.5)         # Inverse cosine
assert bp.arcsin(0.5) == np.arcsin(0.5)         # Inverse sine
assert bp.arctan(1) == np.arctan(1)           # Inverse tangent


