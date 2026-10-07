"""Retained-failure side/XYZ diagnostic; no training or fresh selection here."""
import hashlib,json
from pathlib import Path
from hand3d_v8_common import V7,save
CODE=Path(__file__).resolve().parent;RUN=V7.parent/'hard_side_v16'
RUN.mkdir(exist_ok=True)
source=(CODE/'prepare_side_data_v16.py').read_text();before="V7.parent/'dense_sampling_v13/fresh_rows.json'";after="V7.parent/'hard_dense_v14/fresh_rows.json'"
assert source.count(before)==1;generated=source.replace(before,after).replace("row['set']!='train_development'","row['set']!='retained_failures'")
snapshot=RUN/'assembler_snapshot.py';snapshot.write_text(generated)
save(RUN/'source_provenance.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),scope='47alreadyinspectedfailures; diagnostic only; no training'))
ns=dict(__name__='hard_side_v16_worker',__file__=str(snapshot));exec(compile(generated,str(snapshot),'exec'),ns);ns['RUN']=RUN;ns['SOURCE']=V7.parent/'hard_aligned_v14';ns['main']()
for path in [RUN/'ready.json',RUN/'control/ready.json',RUN/'consensus/ready.json']:
    obj=json.loads(path.read_text());obj['scope']='47retainedfailures; oldcentersRGB/labels exact; no training or independent test claim';save(path,obj)
