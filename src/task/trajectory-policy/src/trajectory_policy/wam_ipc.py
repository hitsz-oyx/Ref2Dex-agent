"""Bounded JSON frames for the two pinned Torch runtimes; no pickle on wire."""
import json
import socket
import struct

SCHEMA = 'ref2dex.wam-native-ipc.v1'
# One batched PPO step is several H4 histories plus several (24, 33) trajectories.
# 4 MiB still rejects an unbounded frame; a 16-env packet is a few hundred KiB.
MAX_BYTES = 4 * 1024 * 1024


def send(connection, packet):
    data = json.dumps(packet, allow_nan=False, separators=(',', ':')).encode()
    if len(data) > MAX_BYTES:
        raise ValueError('IPC frame exceeds limit')
    connection.sendall(struct.pack('!I', len(data)) + data)


def _read(connection, size):
    parts=[]
    while size:
        part=connection.recv(size)
        if not part:
            raise EOFError('IPC peer closed')
        parts.append(part)
        size-=len(part)
    return b''.join(parts)


def receive(connection):
    size=struct.unpack('!I',_read(connection,4))[0]
    if not 0<size<=MAX_BYTES:
        raise ValueError('invalid IPC length')
    packet=json.loads(_read(connection,size))
    if not isinstance(packet,dict):
        raise ValueError('IPC dictionary required')
    return packet


def peer_pid(connection):
    credentials=connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize('3i'))
    return struct.unpack('3i',credentials)[0]
