"""Inspect complete tar members already flushed in partial files, without writing."""
import json,tarfile,time
from pathlib import Path
root=Path('/mnt/why/HOT3D');rows=[]
for p in sorted((root/'rgb_clips').rglob('*.tar.partial')):
    size=p.stat().st_size;offset=0;rgb=0
    with p.open('rb') as f:
        while offset+512<=size:
            f.seek(offset);block=f.read(512)
            if not block or block==b'\0'*512:break
            try:t=tarfile.TarInfo.frombuf(block,'utf-8','surrogateescape')
            except tarfile.HeaderError:break
            end=offset+512+((t.size+511)//512)*512
            if end>size:break
            if t.name.endswith('.image_214-1.jpg'):rgb+=1
            offset=end
    rows.append(dict(clip=p.stem,flushed_rgb_frames=rgb,expected_frames=150,
                     bytes=size,seconds_since_flush=round(time.time()-p.stat().st_mtime)))
print(json.dumps(rows,indent=2))
