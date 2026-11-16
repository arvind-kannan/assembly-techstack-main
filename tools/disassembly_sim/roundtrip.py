"""Round trip against DisassemblySim: get_state, load_assembly, look_at_part.

Same wire protocol as dasg_hmi.py (pickled dict, 16-byte size, ACK), but reads
the reply in a loop so large look_at_part frames arrive whole.
"""

import pickle
import socket
import sys

import numpy as np

HOST, PORT = "127.0.0.1", 12345
XML = sys.argv[1]
PART = sys.argv[2] if len(sys.argv) > 2 else "BG_Tellerrad_V1p"


def recv_exact(s, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"closed after {len(buf)}/{n} bytes")
        buf += chunk
    return bytes(buf)


def request(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(600)
        s.connect((HOST, PORT))
        s.sendall(pickle.dumps(data))
        size = int.from_bytes(recv_exact(s, 16), "big")
        s.sendall(b"ACK")
        return pickle.loads(recv_exact(s, size)), size


print("get_state ->", request({"request": "get_state"}))
print(
    "load_assembly ->",
    request({"request": "load_assembly", "file": XML, "use_loading_point": False}),
)
answer, size = request(
    {
        "request": "look_at_part",
        "part_name": PART,
        "distance": 0.5,
        "theta": 30 / 180 * np.pi,
        "phi": -90 / 180 * np.pi,
    }
)
frame = answer["frame"]
print(f"look_at_part -> state={answer['state']!r}, {size} bytes, keys={sorted(frame)}")
for key in ("points", "index_array", "view_matrix"):
    value = np.asarray(frame[key])
    print(f"  {key}: shape={value.shape} dtype={value.dtype}")
print("  inverted_dict:", frame.get("inverted_dict"))
idx = np.asarray(frame["index_array"]).ravel()
print("  visible body ids:", np.unique(idx[idx >= 0]).tolist())
