import http.client
import json
import threading

from livecap.web import CaptionServer


def test_page_and_sse_roundtrip():
    srv = CaptionServer(port=0)
    try:
        c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
        c.request("GET", "/")
        r = c.getresponse()
        body = r.read().decode()
        assert r.status == 200 and "EventSource('/events')" in body
        assert "http://" not in body and "https://" not in body  # no external assets
        c.close()

        got = {}

        def listen():
            s = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
            s.request("GET", "/events")
            resp = s.getresponse()
            got["line"] = resp.fp.readline().decode()
            s.close()

        t = threading.Thread(target=listen)
        t.start()
        for _ in range(50):  # wait for the subscriber to register
            if srv.subscribers:
                break
            threading.Event().wait(0.02)
        srv.publish("hello captions")
        t.join(5)
        assert json.loads(got["line"][len("data: "):]) == {"text": "hello captions"}
    finally:
        srv.close()
