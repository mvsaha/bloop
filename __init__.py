import numpy as np


try:
    import numba
    nopython_wrapper = numba.njit
except (ImportError):
    nopython_wrapper = lambda x: x

    
def to_array_if_not(x):
    """Convert a value to a numpy array if it isn't already one."""
    return np.array(x) if type(x) is not np.ndarray else x


def maybe_to_scalar(x):
    """Convert a value to a python scalar type if it can be."""
    return np.asscalar(x) if x.size == 1 else x


def repeat(x):
    """Yield a value infinitely."""
    while True:
        yield x


def expand_shape(seq, l, fill_value=None):
    """Expand a sequence with a fill value """
    seq = tuple(seq)
    if len(seq) > l:
       return seq
    seq = ((l - len(seq)) * (fill_value,)) + seq
    return seq


def nest_functions(a, b):
    """Usher output from function a into function b."""
    if a is None:
        return b
    else:
        def nested(*inputs):
            return b(*a(*inputs))
    return nopython_wrapper(nested)


def last_result(a):
    """Wrap a function so that it only returns the last output."""
    def ret(*inputs):
        return a(*inputs)[-1]
    return nopython_wrapper(ret)


def build_extractor(axis_shape, _depth=0, _prev_extract=None):
    """Recursively build a function that extracts an values at an
    ndindex from a ndarray."""
    if _prev_extract is None:
        _depth = 0 if axis_shape is None else len(axis_shape) - 1
    
    if axis_shape is None:
        @nopython_wrapper
        def extract(idx, arr):
            return idx[1:], arr
    
    elif axis_shape[0] is None:
        # Non expanded minimal dimension
        @nopython_wrapper
        def extract(idx, arr):
            return idx[1:], arr
    
    elif axis_shape[0] == 0:
        @nopython_wrapper
        def extract(idx, arr):
            return idx[1:], arr[0]
    else:
        @nopython_wrapper
        def extract(idx, arr):
            return idx[1:], arr[idx[0]]
    
    nested = nest_functions(_prev_extract, extract)
    if _depth == 0:
        return last_result(nested)
    else:
        return build_extractor(axis_shape[1:], _depth - 1, nested)


def extract_along_higher_dims(*args, depth=None, with_index=True):
    """Generate the indices into variables to be separated into last .
    
    Arguments
    ---------
    args : list of scalars or arrays
        Variables that are scalars or have broadcastable dimensions
        along the first `depth` arrays.
    depth : [None] | int
        Integer indicating the dimension to which we should partition
        the args.
    with_index : [True] | False
        If True, then the np.ndindex that was used to index the arrays
        along the first dimension will be returned with the extracted
        values.
    
    Yields
    ------
    If `with_index` is False:
        A tuple the same length of args, consisting of the higher
        dimension values of each input argument.
    If `with_index` is True:
        A len(2) tuple containing the np.ndindex used to extract from
        the lower dimensions as the first element, and the extracted
        values as the second element.
    """
    args = tuple(map(maybe_to_scalar, map(to_array_if_not, args)))
    raw_shapes = tuple(list(np.shape(a)) for a in args)
    ndim = max(map(len, raw_shapes))
    if depth is None:
        depth = ndim
    
    expanded = tuple(map(expand_shape, raw_shapes, repeat(ndim)))
    expanded = [None if ex is None else ex[:depth] for ex in expanded]
    
    shp = []
    for dim_num, dim_shapes in enumerate(zip(*expanded)):
        uniq = set(i for i in dim_shapes if i is not None) - {1}
        if len(uniq) is 0:
            shp.append(1)
        elif len(uniq) is 1:
            shp.append(next(iter(uniq)))
        else:
            raise ValueError('conflicting shape along dimension {}, received '
              'values {}'.format(dim_num, uniq))
    
    corrected = []
    for ex in expanded:
        if all(e == None for e in ex):
            corrected.append(None)
        else:
            _corr = []
            for i, v in enumerate(ex):
                if v == 1 and shp[i] != v:
                    _corr.append(0)  # Consume idx[0] and pass arr[0]
                elif v is None:
                    _corr.append(None)  # Consume idx[0] and pass arr
                else:
                    _corr.append(v)  # Consume idx[0] pass arr[idx[0]]
            corrected.append(_corr)
    
    extractors = [build_extractor(corr) for corr in corrected]
    
    if with_index:
        for idx in np.ndindex(*shp[:depth]):
            yield idx, tuple(ex(idx, a) for ex, a in zip(extractors, args))
    else:
        for idx in np.ndindex(*shp[:depth]):
            yield tuple(ex(idx, a) for ex, a in zip(extractors, args))