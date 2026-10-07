from pathlib import Path
import tarfile,json
P=Path('/mnt/why/HOT3D/provenance');payload=(P/'range_prefix.bin').read_bytes()
for offset in [1024,11264,606208]:
    t=tarfile.TarInfo.frombuf(payload[offset:offset+512],'utf-8','surrogateescape');d=json.loads(payload[offset+512:offset+512+t.size]);(P/t.name).write_text(json.dumps(d,indent=2));print(t.name,json.dumps(d)[:6500])
