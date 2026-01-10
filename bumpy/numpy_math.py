import builtins, math
from .numpy_array import ndarray, get_array, array
from .numpy_type_constant import pi

# https://numpy.org/doc/stable/reference/generated/numpy.dot.html
def dot(a, b):
    a = get_array(a)
    return a.dot(b)

# https://numpy.org/doc/stable/reference/generated/numpy.add.html
def add(a,b): return a+b
# https://numpy.org/doc/stable/reference/generated/numpy.subtract.html
def subtract(a,b): return a-b
# https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
def multiply(a,b): return a*b
# https://numpy.org/doc/stable/reference/generated/numpy.divide.html
def divide(a,b): return a/b
# https://numpy.org/doc/stable/reference/generated/numpy.power.html
def power(a,b): return a**b
# https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
def sqrt(a): return a**0.5
# https://numpy.org/doc/stable/reference/generated/numpy.prod.html
def prod(arr):
    result = 1
    for x in arr:
        result *= x
    return result
# https://numpy.org/doc/stable/reference/generated/numpy.round.html
def round(arr, decimals=0):
    arr = get_array(arr)
    if isinstance(arr, ndarray):
        return arr.round(decimals = decimals)
    else:
        return builtins.round(arr, decimals)
# https://numpy.org/doc/stable/reference/generated/numpy.max.html
def max(*arr): 
    arr = get_array(arr)
    if isinstance(arr, ndarray):
        return arr.max()
    else:
        return builtins.max(arr)
# https://numpy.org/doc/stable/reference/generated/numpy.min.html
def min(*arr): 
    arr = get_array(arr)
    if isinstance(arr, ndarray):
        return arr.min()
    else:
        return builtins.min(arr)
# https://numpy.org/doc/stable/reference/generated/numpy.sum.html
def sum(arr): 
    arr = get_array(arr)
    if isinstance(arr, ndarray):
        return arr.sum()
    else:
        return builtins.sum(arr)
# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
def mean(arr):
    arr = get_array(arr)
    if isinstance(arr, ndarray):
        return arr.mean()
    else:
        return builtins.mean(arr)
    
def _unary_op(arr,op):
    arr = get_array(arr)
    if isinstance(arr, ndarray):
        return arr._unary_op(op)
    else:
        return op(arr)
# https://numpy.org/doc/stable/reference/generated/numpy.floor.html
def floor(a): return _unary_op(a, math.floor)
# https://numpy.org/doc/stable/reference/generated/numpy.ceil.html
def ceil(a): return _unary_op(a, math.ceil)
# https://numpy.org/doc/stable/reference/generated/numpy.sin.html
def sin(a): return _unary_op(a, math.sin)
# https://numpy.org/doc/stable/reference/generated/numpy.cos.html
def cos(a): return _unary_op(a, math.cos)
# https://numpy.org/doc/stable/reference/generated/numpy.tan.html
def tan(a): return _unary_op(a, math.tan)
# https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html
def arcsin(a): return _unary_op(a, math.asin)
# https://numpy.org/doc/stable/reference/generated/numpy.arccos.html
def arccos(a): return _unary_op(a, math.acos)
# https://numpy.org/doc/stable/reference/generated/numpy.arctan.html
def arctan(a): return _unary_op(a, math.atan)
# https://numpy.org/doc/stable/reference/generated/numpy.sinh.html
def sinh(a): return _unary_op(a, math.sinh)
# https://numpy.org/doc/stable/reference/generated/numpy.cosh.html
def cosh(a): return _unary_op(a, math.cosh)
# https://numpy.org/doc/stable/reference/generated/numpy.tanh.html
def tanh(a): return _unary_op(a, math.tanh)
# https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html
def arcsinh(a): return _unary_op(a, math.asinh)
# https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html
def arccosh(a): return _unary_op(a, math.acosh)
# https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html
def arctanh(a): return _unary_op(a, math.atanh)
# https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html
def deg2rad(a): return a*pi/180
# https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html
def rad2deg(a): return a*180/pi
# https://numpy.org/doc/stable/reference/generated/numpy.median.html
def median(arr):
    s = sorted(arr)
    n = len(s)
    mid = n//2
    if n%2==0:
        return (s[mid-1]+s[mid])/2
    else:
        return s[mid]
# https://numpy.org/doc/stable/reference/generated/numpy.std.html
def std(arr):
    arr = get_array(arr)
    return arr.std()
# https://numpy.org/doc/stable/reference/generated/numpy.var.html
def var(arr):
    arr = get_array(arr)
    return arr.var()
# https://numpy.org/doc/stable/reference/generated/numpy.gcd.html
def gcd(a,b):
    while b: a,b = b, a%b
    return a
# https://numpy.org/doc/stable/reference/generated/numpy.lcm.html
def lcm(a,b):
    return abs(a*b)//gcd(a,b)


# https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
def gradient(arr):
    # simple finite difference gradient
    data = arr
    grad = []
    n = len(data)
    for i in range(n):
        if i==0:
            grad.append(data[1]-data[0])
        elif i==n-1:
            grad.append(data[-1]-data[-2])
        else:
            grad.append((data[i+1]-data[i-1])/2)
    return array(grad)

# https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html
def trapezoid(y, x=None, dx=1.0):
    """
    Simple trapezoidal integration.
    y: list or ndarray
    x: list or ndarray, optional
    dx: spacing between points if x is None
    """
    ydata = y
    
    if x is None:
        xdata = [i*dx for i in range(len(ydata))]
    else:
        xdata = x
    
    s = 0
    for i in range(len(ydata)-1):
        s += (xdata[i+1]-xdata[i]) * (ydata[i+1]+ydata[i]) / 2
    return s

# https://numpy.org/doc/stable/reference/generated/numpy.binary_repr.html
def binary_repr(num, width=None):
    """
    Return the binary representation of an integer as a string.
    If width is specified, pad with zeros on the left.
    """
    if num < 0:
        raise ValueError("Only non-negative integers supported")
    s = bin(num)[2:]  # remove '0b' prefix
    if width is not None:
        s = s.zfill(width)
    return s


# https://numpy.org/doc/stable/reference/generated/numpy.hypot.html
def hypot(x,y):
    return math.sqrt(x**2+y**2)