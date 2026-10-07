import json,subprocess
from pathlib import Path
root=Path('/mnt/why/HOT3D');path=root/'completion_status.json'
data=json.loads(path.read_text());assert data['stage']=='complete'
data['storage']=''.join(subprocess.check_output(['du','-sh',str(p)],text=True) for p in [root/'rgb_clips',root/'export',root])
data['allocated_storage_bytes']=int(subprocess.check_output(['du','-s','-B1',str(root)],text=True).split()[0])
part=path.with_suffix('.json.partial');part.write_text(json.dumps(data,indent=2));part.replace(path)
print(json.dumps(data,indent=2))
