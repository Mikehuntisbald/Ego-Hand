import json
from hand3d_v8_common import V7,save
RUN=V7.parent/'fourth_dense_v14';manifest=json.loads((RUN/'fresh_manifest.json').read_text());inventory=json.loads((RUN/'expanded_inventory.json').read_text());lookup={r['sequence']:r for r in inventory['rows']};bad=[]
for r in manifest['sequences']:
    for cid in r['clips']:
        if cid not in lookup[r['sequence']]['unused']:bad.append(dict(sequence=r['sequence'],clip=cid))
save(RUN/'unseen_check.json',dict(passed=not bad,previously_seen=bad,scope='Exported filenames plus all explicit frame/observation/clip manifests and original used inventory; no metric values used'))
assert not bad,bad
print((RUN/'unseen_check.json').read_text())
