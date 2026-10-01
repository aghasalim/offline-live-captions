"""Minimal local caption page: one HTML file, stdlib HTTP server, SSE stream.

Binds to 127.0.0.1 only. The page shows large high-contrast captions, keeps the
full transcript in the browser, and lets the user search it or save it as a
text file. No external assets are loaded.
"""
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Live captions</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#000;--fg:#fff;--dim:#aaa;--acc:#ffd400}
body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif}
#bar{display:flex;gap:8px;padding:8px 16px;align-items:center;background:#111;position:sticky;top:0}
#bar input,#bar button{font-size:16px;padding:6px 10px;border-radius:6px;border:1px solid #444;background:#222;color:var(--fg)}
#bar input{flex:1}
#live{font-size:48px;line-height:1.3;padding:24px 16px;min-height:3.9em;border-bottom:2px solid #333}
#log{padding:16px;font-size:24px;line-height:1.5;color:var(--dim)}
#log p{margin:0 0 .5em}
#log mark{background:var(--acc);color:#000}
#status{color:var(--dim);font-size:14px}
</style></head><body>
<div id="bar">
  <input id="q" placeholder="search transcript" aria-label="search transcript">
  <button id="save">Save .txt</button>
  <label><input type="checkbox" id="big"> bigger</label>
  <span id="status">connecting</span>
</div>
<div id="live" aria-live="polite"></div>
<div id="log"></div>
<script>
const live=document.getElementById('live'),log=document.getElementById('log'),
      q=document.getElementById('q'),st=document.getElementById('status');
const lines=[];
function render(){
  const needle=q.value.trim().toLowerCase();
  log.innerHTML='';
  for(const t of lines){
    if(needle&&!t.toLowerCase().includes(needle))continue;
    const p=document.createElement('p');
    if(needle){const i=t.toLowerCase().indexOf(needle);
      p.append(t.slice(0,i));const m=document.createElement('mark');m.textContent=t.slice(i,i+needle.length);p.append(m,t.slice(i+needle.length));}
    else p.textContent=t;
    log.append(p);
  }
}
q.oninput=render;
document.getElementById('big').onchange=e=>{live.style.fontSize=e.target.checked?'72px':'48px'};
document.getElementById('save').onclick=()=>{
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([lines.join('\\n')],{type:'text/plain'}));
  a.download='transcript.txt';a.click();
};
const es=new EventSource('/events');
es.onopen=()=>st.textContent='listening';
es.onerror=()=>st.textContent='disconnected';
es.onmessage=ev=>{
  const d=JSON.parse(ev.data);
  if(!d.text)return;
  lines.push(d.text);live.textContent=lines.slice(-2).join(' ');render();
  window.scrollTo(0,document.body.scrollHeight);
};
</script></body></html>
"""


class CaptionServer:
    def __init__(self, host="127.0.0.1", port=8765):
        self.subscribers = []
        self.lock = threading.Lock()
        page = PAGE
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    q = queue.Queue()
                    with srv.lock:
                        srv.subscribers.append(q)
                    try:
                        while True:
                            msg = q.get()
                            self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        with srv.lock:
                            srv.subscribers.remove(q)
                    return
                body = page.encode()
                self.send_response(200 if self.path == "/" else 404)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer((host, port), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def publish(self, text):
        with self.lock:
            for q in self.subscribers:
                q.put({"text": text})

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
