import builtins
from .numpy_type_constant import (
    Int64,
    Float64,
    Bool,
    Object,
    DValue,
    DType,
    _dtype_promotion,
    transform_type,
)

def _product(*sequences):
    if not sequences:
        return [()]
    else:
        result = []
        first, rest = sequences[0], sequences[1:]
        rest_product = _product(*rest)
        for item in first:
            for prod in rest_product:
                result.append((item,) + prod)
        return result


def _ndarray_parts_from_object(o):
    
    def inhomogeneous():
        raise TypeError('the requested array has inhomogeneous shape')

    def walk(o, axis):
        nonlocal result_dtype
        t = type(o)
        if t is str or t is dict or t is set:
            # special case when o is one of these iterable types
            if axis != len(result_shape): inhomogeneous()
            result_elems.append(o)
            result_dtype = object
            return
        try:
            elems = list(o)
        except TypeError as exc:
            # happens when o is not an iterable type
            if axis != len(result_shape): inhomogeneous()
            result_elems.append(o)
            if result_dtype is None:
                result_dtype = t
            elif t is not result_dtype:
                if t is float:
                    if result_dtype is int or result_dtype is bool:
                        result_dtype = float
                    else:
                        result_dtype = object
                elif t is int:
                    if result_dtype is bool:
                        result_dtype = int
                    elif result_dtype is not float:
                        result_dtype = object
                elif t is bool:
                    if not (result_dtype is float or result_dtype is int):
                        result_dtype = object
                else:
                    result_dtype = object
            return
        n = len(elems)
        if len(result_shape) == axis:
            result_shape.append(n)
        elif result_shape[axis] != n:
            inhomogeneous()
        axis += 1
        i = 0
        while i < n:
            walk(elems[i], axis)
            i += 1

    result_shape = []
    result_dtype = None
    result_elems = []

    walk(o, 0)

    if result_dtype is None or result_dtype is float:
        result_dtype = Float64
    elif result_dtype is int:
        result_dtype = Int64
    elif result_dtype is bool:
        result_dtype = Bool
    else:
        result_dtype = Object

    return tuple(result_shape), result_dtype, result_elems


def _size_from_shape(shape):
    i = len(shape)-1
    if i == -1: return 1
    size = shape[i]
    while i > 0:
        i -= 1
        size *= shape[i]
    return size

def _strides_from_shape(shape):
    i = len(shape)
    strides = [1] * i
    i -= 1
    size = shape[i]
    while i > 0:
        i -= 1
        strides[i] = size
        size *= shape[i]
    return tuple(strides)

def _flat_index_from_multi(idx_tuple, shape):
    # row‐major flatten of multi‐index into single integer
    flat = 0
    stride = 1
    for i in reversed(range(len(shape))):
        flat += idx_tuple[i] * stride
        stride *= shape[i]
    return flat

def _flatten_nested_list(lst):
    """Flatten nested Python lists into a single list of scalars."""
    flat = []
    for x in lst:
        if isinstance(x, list):
            flat.extend(_flatten_nested_list(x))
        else:
            flat.append(x)
    return flat

def _broadcast_shape(shape1, shape2):
    """Compute the broadcasted shape from two shapes, or raise ValueError."""
    result = []
    for s1, s2 in zip(shape1[::-1], shape2[::-1]):
        if s1 == s2:
            result.append(s1)
        elif s1 == 1:
            result.append(s2)
        elif s2 == 1:
            result.append(s1)
        else:
            raise ValueError(f"shapes {shape1} and {shape2} not compatible for\
                               broadcasting")
    # Append remaining leading dims from the longer shape
    if len(shape1) > len(shape2):
        result.extend(shape1[:len(shape1)-len(shape2)][::-1])
    else:
        result.extend(shape2[:len(shape2)-len(shape1)][::-1])
    return tuple(result[::-1])

def _broadcast_strides(shape, strides, target_shape):
    """
    Given shape & strides of an array, calculate new strides for broadcasting
    to target_shape.
    """
    ndim_diff = len(target_shape) - len(shape)
    new_strides = [0] * len(target_shape)
    # Align from the right
    for i in range(len(shape)):
        if shape[-1 - i] == target_shape[-1 - i]:
            new_strides[-1 - i] = strides[-1 - i]
        elif shape[-1 - i] == 1:
            new_strides[-1 - i] = 0
        else:
            raise ValueError("broadcasting error")
    # Leading dims get stride 0 (broadcasted)
    for i in range(ndim_diff):
        new_strides[i] = 0
    return tuple(new_strides)

def get_array(other, singleton = False):
    if isinstance(other, ndarray):
        return other
    if isinstance(other, list):
        return array(other)
    if isinstance(other, tuple):
        return array(other)
    if isinstance(other, dict):
        return other
    return array([other]) if singleton else other

# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
class ndarray(object):
    def __init__(self, shape, dtype='float64', buffer=None, offset=0,
                 strides=None, order=None):
        shape = tuple(shape)
        if strides is None:
            strides = _strides_from_shape(shape)
        self._shape = shape
        self._size = _size_from_shape(shape)
        if isinstance(dtype, str):
            if dtype == 'float64':
                dtype = Float64
            elif dtype == 'int64':
                dtype = Int64
            elif dtype == 'bool':
                dtype = Bool
            elif dtype == 'O' or dtype == 'object':
                dtype = Object
            else:
                raise TypeError(f"Unsupported dtype string: {dtype}")
        self._dtype = dtype

        if buffer is None:
            # Initialize default buffer with zero values for the dtype
            if self._dtype in [Float64, Int64, Bool]:
                buffer = [DValue(0, self._dtype)] * self._size
            else:
                buffer = [DValue(None, self._dtype)] * self._size
        else:
            self._verify_type(buffer)
        
        self._buffer = buffer
        self._offset = offset
        self._strides = strides

    def _verify_type(self, buffer):
        def rec(buf):
            if not isinstance(buf, list):
                return
            for i in range(len(buf)):
                if isinstance(buf[i], ndarray):
                    rec(buf[i]._buffer)
                elif isinstance(buf[i], list):
                    rec(buf[i])
                elif isinstance(buf[i], tuple):
                    rec(buf[i])
                elif isinstance(buf[i], DValue):
                    pass
                else:
                    buf[i] = DValue(buf[i], self._dtype)
        rec(buffer)

    # https://numpy.org/doc/stable/user/basics.indexing.html
    def _fancy_indexing(self, indices, axes, is_setter, value=None):
        index_lists = []
        for ind in indices:
            if isinstance(ind, ndarray):
                if ind._dtype != 'int64':
                    raise IndexError("Fancy indexing only supports integer\
                                      arrays")
                index_lists.append(ind._buffer[ind._offset: ind._offset +
                                               ind._size])
            else:
                index_lists.append(ind)

        zip_mode = (
            len(index_lists) >= 2 and
            len(index_lists) == len(axes) == len(set(axes)) and
            all(len(index_lists[0]) == len(lst) for lst in index_lists)
        )

        fancy_axes_set = set(axes)
        non_fancy_axes = []
        for i in range(self.ndim):
            if i not in fancy_axes_set:
                non_fancy_axes.append(i)

        if zip_mode:
            combos = []
            for i in range(len(index_lists[0])):
                combo = {}
                for ax, lst in zip(axes, index_lists):
                    combo[ax] = lst[i]
                combos.append(combo)
        else:
            combos = []
            for tup in _product(*index_lists):
                combo = {}
                for ax, val in zip(axes, tup):
                    combo[ax] = val
                combos.append(combo)

        result_shape = [self._shape[ax] for ax in non_fancy_axes]
        if zip_mode:
            result_shape.append(len(index_lists[0]))
        else:
            result_shape.append(len(combos))

        if is_setter:
            if isinstance(value, list):
                value = array(value)
            elif not isinstance(value, ndarray):
                # broadcast scalar
                flat_val = [value] * _size_from_shape(result_shape)
                value = ndarray(result_shape, self._dtype, flat_val)
            elif value.shape != tuple(result_shape):
                raise ValueError("Shape mismatch in fancy indexing assignment")
        else:
            result = ndarray(result_shape, self._dtype)

        for base in _product(*(range(self._shape[ax]) for ax in non_fancy_axes)):
            for combo_idx, combo in enumerate(combos):
                full_index = [0] * self.ndim
                for ax, lst in zip(axes, index_lists):
                    full_index[ax] = lst[combo_idx] if zip_mode else combo[ax]
                for j, ax in enumerate(non_fancy_axes):
                    full_index[ax] = base[j]

                list_off = []
                for k in range(self.ndim):
                    list_off.append(full_index[k] * self._strides[k])
                off = self._offset + sum(list_off)


                if zip_mode:
                    res_idx = (combo_idx,) + base
                else:
                    res_idx = base + (combo_idx,)
                   
                if is_setter:
                    list_src = []
                    for k in range(value.ndim):
                        list_src.append(res_idx[k] * value._strides[k])
                    src_off = value._offset + sum(list_src)
                    self._buffer[off] = value._buffer[src_off]
                else:
                    list_res = []
                    for k in range(result.ndim):
                        list_res.append(res_idx[k] * result._strides[k])
                    res_off = result._offset + sum(list_res)
                    result._buffer[res_off] = self._buffer[off]

        if not is_setter:
            return result


    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__setitem__.html
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        # Pad index with full slices if shorter than ndim
        if len(index) < self.ndim:
            index = index + (slice(None),) * (self.ndim - len(index))

        offset = self._offset
        shape = []
        strides = []

        fancy_indices = []
        fancy_axes = []

        for axis, ind in enumerate(index):
            if isinstance(ind, int):
                if ind < 0:
                    ind += self._shape[axis]
                if ind >= self._shape[axis] or ind < 0:
                    raise IndexError("index out of range")
                offset += ind * self._strides[axis]
            elif isinstance(ind, slice):
                start, stop, step = ind.indices(self._shape[axis])
                max_test = (stop - start + step + (-1 if step > 0 else 1))//step
                length = max(0, max_test)
                shape.append(length)
                strides.append(self._strides[axis] * step)
                offset += self._strides[axis] * start
            elif (isinstance(ind, list) or
                  (isinstance(ind, ndarray) and ind.dtype == 'int64')):
                fancy_indices.append(ind)
                fancy_axes.append(axis)
                # Fancy indexing handled later
            else:
                raise IndexError(f"unsupported index type {type(ind)}")

        if fancy_indices:
            # Check if all fancy indices have same length AND cover different axes
            return self._fancy_indexing(fancy_indices, fancy_axes, False, None)
        else:
            if len(shape) == 0:
                # Return scalar value
                return self._buffer[offset]
            return ndarray(
                tuple(shape),
                self._dtype,
                self._buffer,
                offset,
                tuple(strides),
            )

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__setitem__.html
    def __setitem__(self, index, value):
        self._verify_type(value)
        
        if not isinstance(index, tuple):
            index = (index,)

        if len(index) < self.ndim:
            index = index + (slice(None),) * (self.ndim - len(index))

        offset = self._offset
        shape = []
        strides = []

        fancy_indices = []
        fancy_axes = []

        for axis, ind in enumerate(index):
            if isinstance(ind, int):
                if ind < 0:
                    ind += self._shape[axis]
                if ind >= self._shape[axis] or ind < 0:
                    raise IndexError("index out of range")
                offset += self._strides[axis] * ind
            elif isinstance(ind, slice):
                start, stop, step = ind.indices(self._shape[axis])
                max_test = (stop - start + step + (-1 if step > 0 else 1))//step
                length = max(0, max_test)
                shape.append(length)
                strides.append(self._strides[axis] * step)
                offset += self._strides[axis] * start
            elif (isinstance(ind, list) or
                  (isinstance(ind, ndarray) and ind.dtype == 'int64')):
                fancy_indices.append(ind)
                fancy_axes.append(axis)
            else:
                raise IndexError(f"unsupported index type {type(ind)}")

        if fancy_indices:
            self._fancy_indexing(fancy_indices, fancy_axes, True, value)
        else:
            if len(shape) == 0:
                # Scalar assignment
                self._buffer[offset] = value
            else:
                # Assigning to a slice
                target = ndarray(
                    tuple(shape),
                    self._dtype,
                    self._buffer,
                    offset,
                    tuple(strides)
                )
                if isinstance(value, ndarray):
                    if target.shape != value.shape:
                        raise ValueError("shape mismatch in assignment")
                    def recurse(idx, axis):
                        if axis == len(target.shape):
                            dst_list = []
                            for i, s in zip(idx, target._strides):
                                dst_list.append(i * s)
                            src_list = []
                            for i, s in zip(idx, value._strides):
                                src_list.append(i * s)
                            dst_offset = target._offset + sum(dst_list)
                            src_offset = value._offset + sum(src_list)
                            target._buffer[dst_offset] = value._buffer[src_offset]
                        else:
                            for i in range(target.shape[axis]):
                                recurse(idx + (i,), axis + 1)
                    recurse((), 0)

                elif isinstance(value, list):
                    flat = _flatten_nested_list(value)
                    if len(flat) != target.size:
                        raise ValueError("shape mismatch in assignment")
                    def recurse_write(idx, axis, flat_idx):
                        if axis == len(target.shape):
                            dst_list = []
                            for i, s in zip(idx, target._strides):
                                dst_list.append(i * s)
                            offset = target._offset + sum(dst_list)
                            target._buffer[offset] = flat[flat_idx[0]]
                            flat_idx[0] += 1
                        else:
                            for i in range(target.shape[axis]):
                                recurse_write(idx + (i,), axis + 1, flat_idx)
                    recurse_write((), 0, [0])
                else:
                    raise TypeError("value must be ndarray or nested list")



    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.data.html
    @property
    def data(self):
        return self._buffer[self._offset:self._offset + self._size]
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html
    @property
    def ndim(self):
        return len(self._shape)

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
    @property
    def shape(self):
        return self._shape

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.size.html
    @property
    def size(self):
        return self._size # prod(self.shape)

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html
    @property
    def strides(self):
        return self._strides

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dtype.html
    @property
    def dtype(self):
        return self._dtype
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html
    @property
    def T(self):
        return self.transpose()

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
    @property
    def flags(self):
        raise NotImplementedError

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.itemsize.html
    @property
    def itemsize(self):
        raise NotImplementedError

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nbytes.html
    @property
    def nbytes(self):
        raise NotImplementedError

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.base.html
    @property
    def base(self):
        raise NotImplementedError

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.real.html
    @property
    def real(self):
        raise NotImplementedError

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.imag.html
    @property
    def imag(self):
        raise NotImplementedError

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flat.html
    @property
    def flat(self):
        raise NotImplementedError

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html
    @property
    def ctypes(self):
        raise NotImplementedError

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
    def tolist(self):
        """Convert flat buffer + shape into nested Python lists (like NumPy)."""
        def convert(x):
            if isinstance(x, ndarray):
                result = []
                for i in range(len(x)):
                    result.append(convert(x[i]))
                return result
            if isinstance(x, DValue):
                return x.value
            return x
        return convert(self)

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html
    def reshape(self, newshape): # missing args
        # Support -1 in shape (infer dimension)
        newshape = tuple(newshape)
        if -1 in newshape:
            if newshape.count(-1) > 1:
                raise ValueError("can only specify one unknown dimension")
            known_size = 1
            unknown_idx = -1
            for i, d in enumerate(newshape):
                if d != -1:
                    known_size *= d
                else:
                    unknown_idx = i
            if self.size % known_size != 0:
                raise ValueError(f"cannot reshape array of size {self.size}\
                                   into shape {newshape}")
            newshape = list(newshape)
            newshape[unknown_idx] = self.size // known_size
            newshape = tuple(newshape)

        if _size_from_shape(newshape) != self.size:
            raise ValueError(f"cannot reshape array of size {self.size} into\
                               shape {newshape}")

        new_strides = _strides_from_shape(newshape)
        return ndarray(
            newshape,
            self.dtype,
            self._buffer,
            self._offset,
            new_strides,
        )

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    def flatten(self): # missing args
        """Return a 1D copy of the array."""
        flat_data = [self._buffer[self._offset + i] for i in range(self._size)]
        return ndarray((self._size,), self._dtype, flat_data)

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html
    def copy(self): # missing args
        """Return a deep copy of the array."""
        return self.astype(self._dtype)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html
    def astype(self, dtype): # missing args
        """Return a deep copy of the array as a type."""
        copied_data = [self._buffer[self._offset + i] for i in range(self._size)]
        return ndarray(self._shape, dtype, copied_data, 0, self._strides)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html
    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        if set(axes) != set(range(self.ndim)):
            raise ValueError("invalid axes for transpose")
        new_shape = tuple(self._shape[ax] for ax in axes)
        new_strides = tuple(self._strides[ax] for ax in axes)
        return ndarray(
            new_shape,
            self._dtype,
            self._buffer,
            self._offset,
            new_strides,
        )

    # -------------------------------
    # Arithmetic operations
    # -------------------------------
    def _binary_op(self, other, op, new_dtype=None):
        
        other = get_array(other, singleton=True)
        
        try:
            b_shape = _broadcast_shape(self.shape, other.shape)
        except ValueError:
            raise ValueError(f"shapes {self.shape} and {other.shape} not\
                               compatible")
        b_strides_self = _broadcast_strides(self.shape, self.strides, b_shape)
        b_strides_other = _broadcast_strides(other.shape, other.strides, b_shape)

        # Prepare result array
        if new_dtype is None:
            new_dtype = _dtype_promotion(self.dtype, other.dtype)
        result = ndarray(b_shape, new_dtype)
            
        # Iterate all indices in broadcasted shape and apply op
        def rec(i, offset_self, offset_other, offset_res):
            if i == len(b_shape):
                result._buffer[offset_res] = op(
                    self._buffer[offset_self],
                    other._buffer[offset_other]
                )
                return
            for idx in range(b_shape[i]):
                rec(
                    i + 1,
                    offset_self + idx * b_strides_self[i],
                    offset_other + idx * b_strides_other[i],
                    offset_res + idx * result._strides[i]
                )

        rec(0, self._offset, other._offset, result._offset)
        new_dtype = result._get_real_type()
        result._dtype = new_dtype
        result._verify_type(result._buffer)
        return result
    
    def _unary_op(self, op, new_dtype=None):
        # Prepare result array
        if new_dtype is None:
            new_dtype = self.dtype
        result = ndarray(self.shape, new_dtype)
            
        # Iterate all indices in broadcasted shape and apply op
        def rec(i, offset_self, offset_res):
            if i == len(self.shape):
                result._buffer[offset_res] = op(self._buffer[offset_self])
                return
            for idx in range(self.shape[i]):
                rec(
                    i + 1,
                    offset_self + idx * self._strides[i],
                    offset_res + idx * result._strides[i]
                )

        rec(0, self._offset, result._offset)
        dtype = result._get_real_type()
        result._dtype = dtype
        result._verify_type(result._buffer)
        return result
    
    def _get_real_type(self):
        real_type = self.dtype
        for i in self._buffer:
            real_type = _dtype_promotion(real_type, transform_type(i))
        return real_type

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__lt__.html
    def __lt__(self, other):
        return self._binary_op(other, lambda a, b: a < b, new_dtype=Bool)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__le__.html
    def __le__(self, other):
        return self._binary_op(other, lambda a, b: a <= b, new_dtype=Bool)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__gt__.html
    def __gt__(self, other):
        return self._binary_op(other, lambda a, b: a > b, new_dtype=Bool)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__ge__.html
    def __ge__(self, other):
        return self._binary_op(other, lambda a, b: a >= b, new_dtype=Bool)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__neg__.html
    def __neg__(self):
        return self._unary_op(lambda a: -a)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__pos__.html
    def __pos__(self):
        return self._unary_op(lambda a: +a)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__abs__.html
    def __abs__(self):
        return self._unary_op(lambda a: abs(a))
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__invert__.html
    def __invert__(self):
        return self._unary_op(lambda a: ~ a)

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__add__.html
    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)
    def __radd__(self, other):
        return self + other

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__sub__.html
    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)
    def __rsub__(self, other):
        return self - other

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__mul__.html
    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)
    def __rmul__(self, other):
        return self * other

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__truediv__.html
    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)
    def __rtruediv__(self, other):
        return self / other
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__floordiv__.html
    def __floordiv__(self, other):
        return self._binary_op(other, lambda a, b: a // b)
    def __rfloordiv__(self, other):
        return self // other
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__mod__.html
    def __mod__(self, other):
        return self._binary_op(other, lambda a, b: a % b)
    def __rmod__(self, other):
        return self % other

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__divmod__.html
    def __divmod__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__pow__.html
    def __pow__(self, other): # missing args
        return self._binary_op(other, lambda a, b: a ** b)
    def __rpow__(self, other):
        return self ** other
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__lshift__.html
    def __lshift__(self, other):
        return self._binary_op(other, lambda a, b: a << b)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__rshift__.html
    def __rshift__(self, other):
        return self._binary_op(other, lambda a, b: a >> b)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__and__.html
    def __and__(self, other):
        return self._binary_op(other, lambda a, b: a & b)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__or__.html
    def __or__(self, other):
        return self._binary_op(other, lambda a, b: a | b)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__xor__.html
    def __xor__(self, other):
        return self._binary_op(other, lambda a, b: a ^ b)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__iadd__.html
    def __iadd__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__isub__.html
    def __isub__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__imul__.html
    def __imul__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__itruediv__.html
    def __itruediv__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__ifloordiv__.html
    def __ifloordiv__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__imod__.html
    def __imod__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__ipow__.html
    def __ipow__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__ilshift__.html
    def __ilshift__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__irshift__.html
    def __irshift__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__iand__.html
    def __iand__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__ior__.html
    def __ior__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__ixor__.html
    def __ixor__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__matmul__.html
    def __matmul__(self, other):
        return self._binary_op(other, lambda a, b: linalg.matmul(a, b))
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__copy__.html
    def __copy__(self):
        return self.copy() # self.copy(order='K')
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__deepcopy__.html
    def __deepcopy__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__reduce__.html
    def __reduce__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__setstate__.html
    def __setstate__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__array__.html
    def __array__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__array_wrap__.html
    def __array_wrap__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__len__.html
    def __len__(self):        
        return self._shape[0]
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__contains__.html
    def __contains__(self, other):
        raise NotImplementedError

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__int__.html
    def __int__(self):
        if self._size == 1:
            return int(self._buffer[self._offset])
        raise TypeError('only length-1 arrays can be converted to Python scalars')

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__float__.html
    def __float__(self):
        if self._size == 1:
            return float(self._buffer[self._offset])
        raise TypeError('only length-1 arrays can be converted to Python scalars')
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__complex__.html
    def __complex__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__str__.html
    #def __str__(self, other):
    #    raise NotImplementedError

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__repr__.html
    def __repr__(self):
        if self._size > 100:
            return f'array([...], shape={self._shape})'

        return 'array(' + repr(self.tolist()) + ')'
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__class_getitem__.html
    def __class_getitem__(self, other):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__eq__.html
    def __eq__(self, other):
        if isinstance(other, ndarray):
            # Fast checks first
            if self._shape != other._shape:
                return False
            return self.tolist() == other.tolist()
        if isinstance(other, list):
            return self.tolist() == other        
        if "tolist" in dir(other): # for numpy testing
            return self.tolist() == other.tolist()
        
        return False
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__ne__.html
    def __ne__(self, other):
        return not self.__eq__(other)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.sum.html
    def sum(self, axis=None):
        if axis is None:
            # Sum all elements
            total = DValue(0, Int64) # TODO
            for i in range(self._size):
                total += self._buffer[self._offset + i]
            return total

        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"axis {axis} is out of bounds for array of\
                               dimension {self.ndim}")

        # Sum along the given axis
        new_shape = self._shape[:axis] + self._shape[axis+1:]
        result = ndarray(new_shape, self._dtype)

        # Recursive helper to iterate all indices except the axis dimension
        def recurse(idx, dim):
            if dim == self.ndim:
                # At leaf: sum over axis dimension
                s = Int64(0)
                for i in range(self._shape[axis]):
                    offset = self._offset
                    stride = self._strides[axis]
                    for d, ix in enumerate(idx):
                        if d >= axis:
                            offset += self._strides[d+1] * ix
                        else:
                            offset += self._strides[d] * ix
                    s += self._buffer[offset + i * stride]
                # Assign to result buffer
                res_offset = 0
                for d, ix in enumerate(idx):
                    res_offset += result._strides[d] * ix
                result._buffer[res_offset] = s
            else:
                if dim == axis:
                    recurse(idx, dim + 1)
                else:
                    for i in range(self._shape[dim]):
                        recurse(idx + (i,), dim + 1)

        if self.ndim == 1:
            # sum along axis=0 is sum all elements
            return self.sum()

        recurse((), 0)
        return result
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.mean.html
    def mean(self, axis=None): # missing args
        if axis is None:
            total = self.sum()
            return total / self._size
        else:
            s = self.sum(axis=axis)
            shape_axis = self._shape[axis]
            return s / shape_axis
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.std.html
    def std(self): # missing args
        return self.var()**0.5
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.var.html
    def var(self): # missing args
        m = self.mean()
        return (sum((x-m)**2 for x in self)/len(self))
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html
    def max(self, axis=None): # missing args
        if axis is None:
            max_val = self._buffer[self._offset]
            for i in range(1, self._size):
                v = self._buffer[self._offset + i]
                if v > max_val:
                    max_val = v
            return max_val
        else:
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"axis {axis} is out of bounds")
            new_shape = self._shape[:axis] + self._shape[axis+1:]
            result = ndarray(new_shape, self._dtype)

            def recurse(idx, dim):
                if dim == self.ndim:
                    offset = self._offset
                    for d, ix in enumerate(idx):
                        if d >= axis:
                            offset += self._strides[d+1] * ix
                        else:
                            offset += self._strides[d] * ix
                    max_val = self._buffer[offset]
                    for i in range(1, self._shape[axis]):
                        v = self._buffer[offset + i * self._strides[axis]]
                        if v > max_val:
                            max_val = v
                    res_offset = 0
                    for d, ix in enumerate(idx):
                        res_offset += result._strides[d] * ix
                    result._buffer[res_offset] = max_val
                else:
                    if dim == axis:
                        recurse(idx, dim + 1)
                    else:
                        for i in range(self._shape[dim]):
                            recurse(idx + (i,), dim + 1)

            if self.ndim == 1:
                return self.max()
            recurse((), 0)
            return result

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.min.html
    def min(self, axis=None): # missing args
        # Similar to max but for min
        if axis is None:
            min_val = self._buffer[self._offset]
            for i in range(1, self._size):
                v = self._buffer[self._offset + i]
                if v < min_val:
                    min_val = v
            return min_val
        else:
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"axis {axis} is out of bounds")
            new_shape = self._shape[:axis] + self._shape[axis+1:]
            result = ndarray(new_shape, self._dtype)

            def recurse(idx, dim):
                if dim == self.ndim:
                    offset = self._offset
                    for d, ix in enumerate(idx):
                        if d >= axis:
                            offset += self._strides[d+1] * ix
                        else:
                            offset += self._strides[d] * ix
                    min_val = self._buffer[offset]
                    for i in range(1, self._shape[axis]):
                        v = self._buffer[offset + i * self._strides[axis]]
                        if v < min_val:
                            min_val = v
                    res_offset = 0
                    for d, ix in enumerate(idx):
                        res_offset += result._strides[d] * ix
                    result._buffer[res_offset] = min_val
                else:
                    if dim == axis:
                        recurse(idx, dim + 1)
                    else:
                        for i in range(self._shape[dim]):
                            recurse(idx + (i,), dim + 1)

            if self.ndim == 1:
                return self.min()
            recurse((), 0)
            return result
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.all.html
    def all(self): # missing args
        for i in range(self._size):
            if not self._buffer[self._offset + i]:
                return False
        return True

    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.any.html
    def any(self): # missing args
        for i in range(self._size):
            if self._buffer[self._offset + i]:
                return True
        return False
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dot.html
    def dot(self, b):
        b = get_array(b)
        type_promo = _dtype_promotion(self.dtype, b.dtype)

        if self.ndim == 1 and b.ndim == 1:
            # vector dot product
            if self.shape[0] != b.shape[0]:
                raise ValueError("vectors must be same length")
            s = DValue(0, type_promo)
            for i in range(self.shape[0]):
                s += self[i] * b[i]
            return s

        if self.ndim == 2 and b.ndim == 1:
            # matrix-vector product
            if self.shape[1] != b.shape[0]:
                raise ValueError("shapes not aligned for matrix-vector dot")
            result = ndarray((self.shape[0],), _dtype_promotion(self.dtype, b.dtype))
            for i in range(self.shape[0]):
                s = DValue(0, type_promo)
                for j in range(self.shape[1]):
                    s += self[i, j] * b[j]
                result[i] = s
            return result

        if self.ndim == 2 and b.ndim == 2:
            return self.matmul(b)

        raise NotImplementedError("dot product not implemented for given\
                                   dimensions")
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.repeat.html
    def repeat(self, repeats, axis=None):
        if axis is not None:
            if axis < 0:
                axis += self.ndim
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"axis {axis} is out of bounds for array of\
                                   dimension {self.ndim}")

        if axis is None:
            # Flatten self and call repeat along axis 0
            flat = self.reshape((self.size,))
            return repeat(flat, repeats, axis=0)

        axis_len = self.shape[axis]

        # Normalize repeats to list of ints matching axis length
        if isinstance(repeats, int):
            repeats_arr = [repeats] * axis_len
        else:
            repeats_arr = get_array(repeats).tolist()
            print(self, repeats, axis)
            print(repeats_arr, axis_len)
            if len(repeats_arr) != axis_len:
                raise ValueError("repeats array length does not match length of\
                                  axis")

        # Calculate output shape
        out_shape = list(self.shape)
        out_shape[axis] = sum(repeats_arr)

        # For simplicity, assume self._buffer is flat row-major list
        step = 1
        for s in self.shape[axis+1:]:
            step *= s

        outer_blocks = self.size // (self.shape[axis] * step)

        new_buf = []
        buf = self._buffer

        for outer_idx in range(outer_blocks):
            base_outer = outer_idx * self.shape[axis] * step
            for axis_idx in range(axis_len):
                count = repeats_arr[axis_idx]
                start = base_outer + axis_idx * step
                slice_data = buf[start:start+step]
                for _ in range(count):
                    new_buf.extend(slice_data)

        return ndarray(tuple(out_shape), self.dtype, new_buf)

    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.round.html
    def round(self, decimals=0): # missing args
        copied_data = []
        for i in range(self._size):
            copied_data.append(builtins.round(
                self._buffer[self._offset + i],
                decimals
            ))
        return ndarray(self._shape, dtype, copied_data, 0, self._strides)
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argmax.html
    def argmax(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argmin.html
    def argmin(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argpartition.html
    def argpartition(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argsort.html
    def argsort(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.byteswap.html
    def byteswap(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.choose.html
    def choose(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.clip.html
    def clip(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.compress.html
    def compress(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conj.html
    def conj(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conjugate.html
    def conjugate(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.cumprod.html
    def cumprod(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.cumsum.html
    def cumsum(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.diagonal.html
    def diagonal(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dump.html
    def dump(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dumps.html
    def dumps(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.fill.html
    def fill(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.getfield.html
    def getfield(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
    def item(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nonzero.html
    def nonzero(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.partition.html
    def partition(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.prod.html
    def prod(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.put.html
    def put(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ravel.html
    def ravel(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.resize.html
    def resize(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.searchsorted.html
    def searchsorted(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setfield.html
    def setfield(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setflags.html
    def setflags(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.sort.html
    def sort(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.squeeze.html
    def squeeze(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.swapaxes.html
    def swapaxes(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.take.html
    def take(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.to_device.html
    def to_device(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tobytes.html
    def tobytes(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html
    def tofile(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.trace.html
    def trace(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html
    def view(self):
        raise NotImplementedError
    
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.getfield.html
    def getfield(self):
        raise NotImplementedError
    
# https://numpy.org/doc/stable/reference/generated/numpy.array.html
def array(object, dtype=None, *, copy=True, order='K', subok=False,
          ndmin=0, ndmax=0, like=None):
    shape, dtype, elems = _ndarray_parts_from_object(object)
    strides = _strides_from_shape(shape)
    return ndarray(shape, dtype, elems, 0, strides)



# https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
def repeat(a, repeats, axis=None):
    a = get_array(a)
    return a.repeat(repeats, axis=axis)

# https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
def meshgrid(*xi, indexing='xy', sparse=False, copy=True):
    n = len(xi)
    shape = [len(x) for x in xi]

    if indexing not in ('xy', 'ij'):
        raise ValueError("indexing must be 'xy' or 'ij'")

    if indexing == 'xy' and n >= 2:
        shape[0], shape[1] = shape[1], shape[0]

    grids = []
    for i, x in enumerate(xi):
        if x.ndim != 1:
            raise ValueError(f"Input array at position {i} must be 1D")

        grid_shape = [1] * n
        if sparse:
            grid_shape[i] = len(x)
        else:
            grid_shape = shape.copy()

        if indexing == 'xy' and n >= 2:
            # For grids after shape swap, swap dims for first two grids accordingly
            if i == 0:
                grid_shape[0], grid_shape[1] = shape[1], shape[0]
            elif i == 1:
                grid_shape[0], grid_shape[1] = shape[1], shape[0]

        grid = ndarray(grid_shape)
        reshape_shape = [1]*n
        reshape_shape[i] = x.size
        x_reshaped = x.reshape(reshape_shape)

        if sparse:
            grid = x_reshaped
        else:
            grid = x_reshaped
            for dim in range(n):
                if dim == i:
                    continue
                repeats_count = grid_shape[dim]
                grid = repeat(grid, repeats_count, axis=dim)

        grids.append(grid)

    # For 'xy' indexing, transpose first two grids back to swap dims 0 and 1
    if indexing == 'xy' and n >= 2:
        grids[0] = grids[0].transpose((1, 0, *range(2, n)))
        grids[1] = grids[1].transpose((1, 0, *range(2, n)))

    return tuple(grids)

# https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
def zeros(shape):
    """Create an ndarray filled with zeros."""
    return full(shape, 0)

# https://numpy.org/doc/stable/reference/generated/numpy.ones.html
def ones(shape):
    """Create an ndarray filled with ones."""
    return full(shape, 1)

# https://numpy.org/doc/stable/reference/generated/numpy.full.html
def full(shape, fill_value):
    """Create an ndarray filled with a specified value."""
    if isinstance(shape, int):
        shape = (shape,)
    def build(s):
        if len(s) == 1:
            return [fill_value] * s[0]
        return [build(s[1:]) for _ in range(s[0])]
    return array(build(shape))

# https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
def zeros_like(arr):
    return full(arr.shape, 0)

# https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html
def ones_like(arr):
    return full(arr.shape, 1)

# https://numpy.org/doc/stable/reference/generated/numpy.full_like.html
def full_like(arr, fill_value):
    return full(shape(arr.shape), fill_value)

# https://numpy.org/doc/stable/reference/generated/numpy.eye.html
def eye(n, m=None):
    """Identity matrix."""
    if m is None: m=n
    return array([[1 if i==j else 0 for j in range(m)] for i in range(n)])

# https://numpy.org/doc/stable/reference/generated/numpy.arange.html
def arange(start, stop=None, step=1):
    """Return evenly spaced values within a given interval."""
    if stop is None:
        start, stop = 0, start
    values = []
    i = start
    while (step > 0 and i < stop) or (step < 0 and i > stop):
        values.append(i)
        i += step
    return array(values)

# https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
def linspace(start, stop, num=50, endpoint=True, retstep=False):
    """Return evenly spaced numbers over a specified interval."""
    if num <= 0:
        data, step = array([]), 0
    elif num == 1:
        return array([start])
    else:
        if endpoint:
            step = (stop - start) / (num - 1) if num > 1 else 0
        else:
            step = (stop - start) / num
        data = array([start + step*i for i in range(num)])
    
    if retstep:
        return data, step
    return data

# https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
def vstack(arrs):
    """
    Stack arrays vertically (along first axis).
    Works for any dimension.
    """
    # Convert all inputs to ndarray
    arrs = [get_array(arr) for arr in arrs]
    
    # Flatten all to lists
    data_lists = [_unflatten(a.data, a.shape) for a in arrs]
    
    # If first array is 1D, convert all to 2D row vectors
    if len(arrs[0].shape) == 1:
        new_data_lists = []
        for x in data_lists:
            if not isinstance(x[0], list):
                new_data_lists.append([x])
            else:
                new_data_lists.append(x)
        data_lists = new_data_lists
    
    # Concatenate along axis 0
    stacked = []
    for lst in data_lists:
        stacked.extend(lst)
    
    return array(stacked)

# https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
def hstack(arrs):
    """
    Stack arrays horizontally (along last axis).
    Works for any dimension.
    """
    arrs = [get_array(arr) for arr in arrs]
    shapes = [a.shape for a in arrs]
    shapes_0 = [s[0] for s in shapes]
    shapes_l = [len(s) for s in shapes]
    if len(set(shapes_l)) != 1:
        raise ValueError("ValueError: all the input arrays must have same \
                          number of dimensions")
    if len(set(shapes_0)) != 1 and shapes_l[0] != 1:
        raise ValueError("all the input array dimensions except for the \
                          concatenation axis must match exactly, but along \
                          dimension 0")
    
    # Unflatten all arrays
    data_unflat = []
    for a in arrs:
        data_unflat.append(_unflatten(a.data, a.shape))
    
    result = []
    if shapes_l[0] == 1:
        for arr in data_unflat:
            result.extend(arr)
    else:
        # Build result row by row
        for i in range(shapes[0][0]):
            row = []
            for arr, s in zip(data_unflat, shapes):
                # 1D arrays → repeat element
                if len(s) == 1:
                    row.append(arr[i] if i < len(arr) else None)
                else:
                    row.extend(arr[i])
            result.append(row)
    return array(result)

# https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html
def column_stack(arrs):
    hstack_arrs = []
    for arr in arrs:
        nd_arr = get_array(arr)
        if len(nd_arr.shape)==1:
            hstack_arrs.append(nd_arr.reshape((nd_arr.shape[0],1)))
        else:
            hstack_arrs.append(nd_arr)
    return hstack(hstack_arrs)

# https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html
def hsplit(arr, n):
    arr = get_array(arr)
    data=_unflatten(arr.data, arr.shape)
    step=arr.shape[-1] // n
    if arr.shape[-1] % n != 0:
        raise ValueError("Array cannot be evenly split along horizontal axis")
    
    result = []
    for i in range(n):
        def slice_axis(d):
            if isinstance(d[0], list):
                return [row[i*step:(i+1)*step] for row in d]
            return d[i*step:(i+1)*step]
        sliced = slice_axis(data)
        result.append(array(sliced))
    return result

# https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html
def vsplit(arr, n):
    arr = get_array(arr)
    data = _unflatten(arr.data, arr.shape)
    
    step = arr.shape[0] // n
    if arr.shape[0] % n != 0:
        raise ValueError("Array cannot be evenly split along vertical axis")
    return [array(data[i*step:(i+1)*step]) for i in range(n)]

# https://numpy.org/doc/stable/reference/generated/numpy.shape.html
def shape(arr):
    """
    Return the shape of an array or list.
    Works for ndarray or nested Python lists.
    """
    arr = get_array(arr)
    if isinstance(arr, ndarray):
        return arr.shape
    raise ValueError("The entry must be an ndarray")



# https://numpy.org/doc/stable/reference/generated/numpy.where.html
def where(condition, x, y):
    """
    Element-wise selection: if condition[i] is True, select x[i], else y[i].
    Works when x and y are scalars/strings or ndarrays of the same shape as condition.
    """
    # Flatten inputs if they are ndarrays, else repeat scalars
    cond_data = condition
    
    if isinstance(x, ndarray):
        x_data = x.data
    else:
        x_data = [x]*len(cond_data)
    
    if isinstance(y, ndarray):
        y_data = y.data
    else:
        y_data = [y]*len(cond_data)
    
    result = [xi if c else yi for xi, yi, c in zip(x_data, y_data, cond_data)]
    return array(result)



def _unflatten(flat_buffer, shape):
    """
    Convert a flat list into nested lists according to shape (row-major).
    """
    if not shape:  # scalar
        return flat_buffer[0]

    if len(shape) == 1:
        return flat_buffer[:shape[0]]

    size = 1
    for s in shape[1:]:
        size *= s

    result = []
    for i in range(shape[0]):
        start = i * size
        end = (i + 1) * size
        result.append(_unflatten(flat_buffer[start:end], shape[1:]))
    return result
