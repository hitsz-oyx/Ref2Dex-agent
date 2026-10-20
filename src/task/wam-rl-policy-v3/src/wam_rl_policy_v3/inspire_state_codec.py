"""Lossless bounded numeric state inside the existing JSON IPC frame; no pickle."""
import base64
import math
import numpy as np

SCHEMA='ref2dex.inspire-state-base64.v1'
FIELDS=('q','dq','physical','objects','hand','tick')
DTYPES={name:np.dtype(name) for name in ('float32','int32','int64')}
MAX_BYTES=2*1024*1024


def pack_state(state):
    result=dict(state);total=0
    for name in FIELDS:
        array=np.ascontiguousarray(state[name])
        if array.dtype.name not in DTYPES or array.dtype!=DTYPES[array.dtype.name] or not 1<=array.ndim<=3:
            raise ValueError('unsupported native numeric state dtype/shape')
        total+=array.nbytes
        if total>MAX_BYTES:raise ValueError('native numeric state exceeds bound')
        if not np.isfinite(array).all():raise ValueError('nonfinite native numeric state')
        result[name]=dict(dtype=array.dtype.name,shape=list(array.shape),data=base64.b64encode(array.tobytes()).decode('ascii'))
    result['array_transport']=SCHEMA
    return result


def unpack_state(state):
    if 'array_transport' not in state:return state
    if state['array_transport']!=SCHEMA:raise ValueError('unsupported native array transport')
    result=dict(state);total=0
    for name in FIELDS:
        field=state[name];dtype=DTYPES.get(field.get('dtype'));shape=field.get('shape')
        if dtype is None or not isinstance(shape,list) or not 1<=len(shape)<=3 or any(type(n) is not int or n<0 for n in shape):
            raise ValueError('invalid numeric state dtype/shape')
        count=math.prod(shape);total+=count*dtype.itemsize
        if total>MAX_BYTES:raise ValueError('native numeric state exceeds bound')
        data=base64.b64decode(field['data'],validate=True)
        if len(data)!=count*dtype.itemsize:raise ValueError('numeric state byte length mismatch')
        result[name]=np.frombuffer(data,dtype=dtype).reshape(shape).copy()
        if not np.isfinite(result[name]).all():raise ValueError('nonfinite native numeric state')
    # A decoded snapshot can traverse both the collector and observation adapter.
    # Only wire packets carry this marker; retaining it attempts a second decode.
    result.pop('array_transport')
    return result
