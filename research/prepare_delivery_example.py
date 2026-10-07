import json
import os
import shutil
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D')
RUN=Path(os.environ.get('HOT3D_DIT_RUN',str(ROOT/'experiments/dit_wilor_v4')))
out=RUN/'delivery';out.mkdir(exist_ok=True)
rows=json.loads((ROOT/'experiments/dit_wilor_v3/locked_rows.json').read_text())
row=next(r for r in rows if r['matched'])
shutil.copy2(row['image'],out/'example.jpg')
(out/'camera.json').write_text(json.dumps(row['camera'],indent=2))
(out/'example_provenance.json').write_text(json.dumps(dict(image=row['image'],role='development, former v3 test',labels_in_input=False),indent=2))
print(str(out))

