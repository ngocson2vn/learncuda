import json
import subprocess as sp
cmd = "./main"

p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, shell=True, encoding="utf-8")
raw_data = p.stdout.read()
if raw_data is not None and len(raw_data) > 0:
  d = json.loads(raw_data)
  print(json.dumps(d, indent=2))
