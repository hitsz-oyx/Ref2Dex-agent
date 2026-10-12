import os
import socket
import struct

import pytest

from trajectory_policy.wam_ipc import MAX_BYTES, peer_pid, receive, send


def test_native_runtime_wire_preserves_finite_json_and_peer_identity():
    a,b=socket.socketpair()
    try:
        value={'tick':0,'tau':[[.01,-.02,0.]],'elapsed_s':.03}
        send(a,value)
        assert receive(b)==value
        assert peer_pid(b)==os.getpid()
        with pytest.raises(ValueError):
            send(a,{'nonfinite':float('nan')})
        with pytest.raises(ValueError):
            send(a,{'large':'a'*MAX_BYTES})
    finally:
        a.close();b.close()


def test_native_runtime_wire_rejects_oversized_header_before_body():
    a,b=socket.socketpair()
    try:
        a.sendall(struct.pack('!I',MAX_BYTES+1))
        with pytest.raises(ValueError):
            receive(b)
    finally:
        a.close();b.close()
