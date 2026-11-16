import pickle
import socket
import threading

import pytest

from sim_server_protocol import InvalidRequest, serve_connection


def dispatch(req):
    if req['request'] == 'echo':
        return {'echo': req['data']}
    if req['request'] == 'boom':
        raise RuntimeError('handler failed')
    if req['request'] == 'bad_fields':
        raise InvalidRequest('missing field')
    if req['request'] == 'missing':
        return {'v': req['required']}
    if req['request'] == 'unpicklable':
        return {'f': lambda: 0}
    raise InvalidRequest(f"unknown request {req['request']!r}")


@pytest.fixture
def client():
    a, b = socket.socketpair()
    t = threading.Thread(target=serve_connection, args=(b, dispatch), daemon=True)
    t.start()
    yield a
    a.close()
    t.join(timeout=5)
    b.close()


def recv_exact(s, n):
    buf = b''
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        assert chunk
        buf += chunk
    return buf


def framed_call(s, obj):
    payload = pickle.dumps(obj) if not isinstance(obj, bytes) else obj
    s.sendall(len(payload).to_bytes(16, 'big') + payload)
    size = int.from_bytes(recv_exact(s, 16), 'big')
    return pickle.loads(recv_exact(s, size))


def legacy_call(s, obj):
    s.sendall(pickle.dumps(obj))
    size = int.from_bytes(recv_exact(s, 16), 'big')
    s.sendall(b'ACK')
    return pickle.loads(recv_exact(s, size))


def test_framed_request_over_256kb_round_trips(client):
    blob = b'x' * (1024 * 1024)
    assert framed_call(client, {'request': 'echo', 'data': blob})['echo'] == blob


def test_legacy_request_with_ack_still_works(client):
    assert legacy_call(client, {'request': 'echo', 'data': 1}) == {'echo': 1}


def test_raising_handler_returns_internal_error_and_server_survives(client):
    r = framed_call(client, {'request': 'boom'})
    assert r['ok'] is False and r['error'] == 'internal_error'
    assert framed_call(client, {'request': 'echo', 'data': 2}) == {'echo': 2}


def test_legacy_raising_handler_does_not_end_server(client):
    assert legacy_call(client, {'request': 'boom'})['error'] == 'internal_error'
    assert legacy_call(client, {'request': 'echo', 'data': 3}) == {'echo': 3}


@pytest.mark.parametrize('req', [
    {'request': 'nope'}, {'request': 'bad_fields'}, {'no_request_key': 1}, {'request': 5}, [1, 2], b'not a pickle',
])
def test_malformed_or_unknown_is_invalid_request(client, req):
    assert framed_call(client, req)['error'] == 'invalid_request'
    assert framed_call(client, {'request': 'echo', 'data': 4}) == {'echo': 4}


def test_missing_field_is_invalid_request(client):
    assert framed_call(client, {'request': 'missing'})['error'] == 'invalid_request'


def test_unpicklable_response_does_not_end_server(client):
    assert framed_call(client, {'request': 'unpicklable'})['error'] == 'internal_error'
    assert framed_call(client, {'request': 'echo', 'data': 5}) == {'echo': 5}
