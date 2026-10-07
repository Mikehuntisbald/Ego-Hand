import requests
from urllib.parse import urlsplit
urls=[
'https://hf-mirror.com/spaces/rolpotamias/WiLoR/resolve/99fe3d7acff8104ecca1055df7467709506c2fa6/pretrained_models/wilor_final.ckpt',
]
for u in urls:
    try:
        with requests.get(u,headers={'Range':'bytes=0-1023'},stream=True,timeout=20) as r:
            print(r.status_code,r.headers.get('Content-Type'),r.headers.get('Content-Length'),urlsplit(r.url).hostname,flush=True)
            print(next(r.iter_content(64))[:20],flush=True)
    except Exception as e:
        print(type(e).__name__,str(e)[:200],flush=True)
