from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
log=Path('mock-posts.jsonl')
seen=set()
class H(BaseHTTPRequestHandler):
 def do_POST(self):
  p=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  with log.open('a') as f: f.write(json.dumps({'path':self.path,'payload':p})+'\n')
  required=['subject_id','session_id','seq','hand','trial_id','label','time_ms']+[f'fsr{i}' for i in range(1,7)]+['ax','ay','az','gx','gy','gz']
  assert all(k in p for k in required)
  key=(p['session_id'],p['hand'],p['seq'])
  code=503 if self.path=='/fail' and p['seq']==2 else 200
  if self.path in ('/saved-then-500','/always-500'):
   code=500 if self.path=='/always-500' or key not in seen else 200
   seen.add(key)
  self.send_response(code);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(b'{"status":"saved"}')
 def log_message(self,*a):pass
HTTPServer(('127.0.0.1',18081),H).serve_forever()
