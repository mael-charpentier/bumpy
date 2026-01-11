# https://numpy.org/doc/stable/reference/arrays.dtypes.html
class DType:
    def __init__(self, name, cast):
        self.name = name
        self.cast = cast
    
    def __call__(self, value):
        #return self.cast(value)
        return DValue(value, self)

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        return False

    def __repr__(self):
        return f"{self.name}"
    
class DValue:
    def __init__(self, value, dtype):
        self.value = dtype.cast(value)
        self.dtype = dtype
        
    def __repr__(self):
        return f"{self.dtype}({self.value})"
    
    def __eq__(self, other):        
        if isinstance(other, DValue):
               return self.value == other.value # and self.dtype == other.dtype
        return self.value == other
    
    def _binary_op(self, other, op, new_dtype=None):
        if isinstance(other, DValue):
            return DValue(op(self.value, other.value),
                          _dtype_promotion(self.dtype, other.dtype)
                            if new_dtype is None else new_dtype)
        return DValue(op(self.value, other),
                      _dtype_promotion(self.dtype, transform_type(other))
                       if new_dtype is None else new_dtype)
    
    def __add__(self, other):
        return self._binary_op(other, lambda x, y: x + y)
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self._binary_op(other, lambda x, y: x - y)
    def __rsub__(self, other):
        return self - other
    
    def __mul__(self, other):
        return self._binary_op(other, lambda x, y: x * y)
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self._binary_op(other, lambda x, y: x / y, new_dtype=Float64)
    def __rtruediv__(self, other):
        return self / other
    
    def __pow__(self, other):
        return self._binary_op(other, lambda x, y: x ** y)
    def __rpow__(self, other):
        return self ** other
    
    def __mod__(self, other):
        return self._binary_op(other, lambda x, y: x % y)
    def __rmod__(self, other):
        return self % other
    
    def __floordiv__(self, other):
        return self._binary_op(other, lambda x, y: x // y)
    def __rfloordiv__(self, other):
        return self // other

    def __and__(self, other):
        return self._binary_op(other, lambda x, y: x & y)

    def __or__(self, other):
        return self._binary_op(other, lambda x, y: x | y)

    def __xor__(self, other):
        return self._binary_op(other, lambda x, y: x ^ y)

    def __lshift__(self, other):
        return self._binary_op(other, lambda x, y: x << y)

    def __rshift__(self, other):
        return self._binary_op(other, lambda x, y: x >> y)

    def __lt__(self, other):
        return self._binary_op(other, lambda x, y: x < y, new_dtype=Bool)
    
    def __le__(self, other):
        return self._binary_op(other, lambda x, y: x <= y, new_dtype=Bool)

    def __gt__(self, other):
        return self._binary_op(other, lambda x, y: x > y, new_dtype=Bool)
    
    def __ge__(self, other):
        return self._binary_op(other, lambda x, y: x >= y, new_dtype=Bool)

    def __neg__(self):
        return DValue(-self.value, self.dtype)

    def __abs__(self):
        return DValue(abs(self.value), self.dtype)

    def __bool__(self):
        return bool(self.value)
    
    def __int__(self):
        return int(self.value)
    
    def __float__(self):
        return float(self.value)
    
    def __index__(self):
        return int(self.value)
    
# https://numpy.org/doc/stable/reference/arrays.promotion.html
def _dtype_promotion(dtype1, dtype2):
    # Simple promotion rules
    if dtype1 == Inf or dtype2 == Inf:
        return Inf
    if dtype1 == Object or dtype2 == Object:
        return Object
    if dtype1 == Float64 or dtype2 == Float64:
        return Float64
    if dtype1 == Int64 and dtype2 == Int64:
        return Int64
    if ((dtype1 == Int64 and dtype2 == Bool) or
        (dtype1 == Bool and dtype2 == Int64)):
        return Int64
    if dtype1 == Bool and dtype2 == Bool:
        return Bool
    # Otherwise fallback
    return Object

def is_instance_by_name(obj, class_name, module_name=None):
    cls = type(obj)
    if cls.__name__ != class_name:
        return False
    if module_name is not None and cls.__module__ != module_name:
        return False
    return True

def transform_type(object_val):
    object_type = type(object_val)
    
    # to not have a cycle of import I test by name
    if is_instance_by_name(object_val, 'ndarray', 'bumpy.numpy_array'):
        return object_type
    
    if object_type == DType:
        return object_val
    if object_type == DValue:
        return object_val.dtype
    if object_type == int:
        return Int64
    if object_type == float:
        return Float64
    if object_type == bool:
        return Bool
    return Object

import math

# Concrete dtypes
Int64 = DType('int64', int)
Float64 = DType('float64', float)
Bool = DType('bool', bool)
Object = DType('object', str)
Inf = DType('inf', lambda x: float('inf'))

# https://numpy.org/doc/stable/reference/constants.html
e = Float64(math.e)
euler_gamma = Float64(0.57721566490153286060651209008240243104215933593992)
pi = Float64(math.pi)
inf = Inf(float('inf'))