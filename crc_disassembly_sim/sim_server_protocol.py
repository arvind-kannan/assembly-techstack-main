"""Wire protocol and error handling of the DisassemblySim server.

Kept free of pybullet so it can be tested on its own.

Two request formats share one socket and are told apart by the first byte:
  * legacy (DASG app): a bare pickle (starts with 0x80, read with one recv, so
    still capped at 256 KB). The response is a 16-byte big-endian length, then
    a 3-byte ACK from the client, then the pickle.
  * framed: a 16-byte big-endian length, then the pickle (read until complete).
    The response is framed the same way, with no ACK.
"""
import pickle

LENGTH_BYTES = 16
LEGACY_RECV_SIZE = 65536 * 4
_PICKLE_MARKER = 0x80


class InvalidRequest(Exception):
    """Raised by a dispatcher for an unknown request or malformed fields."""


def _recv_exact(conn, n):
    chunks = []
    while n:
        chunk = conn.recv(min(n, 1 << 20))
        if not chunk:
            raise ConnectionError('peer closed mid-message')
        chunks.append(chunk)
        n -= len(chunk)
    return b''.join(chunks)


def read_message(conn):
    """Return (payload_bytes, legacy) or None when the client closed the connection."""
    first = conn.recv(1)
    if not first:
        return None
    if first[0] == _PICKLE_MARKER:
        return first + conn.recv(LEGACY_RECV_SIZE), True
    size = int.from_bytes(first + _recv_exact(conn, LENGTH_BYTES - 1), 'big')
    return _recv_exact(conn, size), False


def send_response(conn, data, legacy):
    try:
        payload = pickle.dumps(data)
    except Exception as e:
        payload = pickle.dumps(error_response('internal_error', f'response not serializable: {e}'))
    conn.sendall(len(payload).to_bytes(LENGTH_BYTES, 'big'))
    if legacy:
        if not conn.recv(3):
            print('[send_data] client disconnected before ACK — response dropped.')
            return
    conn.sendall(payload)


def error_response(code, message):
    return {'ok': False, 'state': 'error', 'error': code, 'message': message}


def handle_payload(payload, dispatch):
    """Decode one request and run `dispatch(dict) -> dict`; never raises."""
    try:
        request = pickle.loads(payload)
    except Exception as e:
        return error_response('invalid_request', f'cannot decode request: {e}')
    if not isinstance(request, dict) or not isinstance(request.get('request'), str):
        return error_response('invalid_request', "request must be a dict with a string 'request' field")
    try:
        return dispatch(request)
    except InvalidRequest as e:
        return error_response('invalid_request', str(e))
    except KeyError as e:  # handlers index required fields directly
        return error_response('invalid_request', f'missing field {e}')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return error_response('internal_error', f'{type(e).__name__}: {e}')


def serve_connection(conn, dispatch):
    """Answer requests on one connection until the client closes it."""
    while True:
        try:
            message = read_message(conn)
            if message is None:
                return
            payload, legacy = message
            send_response(conn, handle_payload(payload, dispatch), legacy)
        except (ConnectionError, OSError) as e:
            print(f'[server] connection lost: {e}')
            return
