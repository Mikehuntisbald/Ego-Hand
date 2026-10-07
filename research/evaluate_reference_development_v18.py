"""Reuse the same unchanged recovery/protection gates for reference condition."""
import hashlib
from pathlib import Path
from hand3d_v8_common import V7,save
CODE=Path(__file__).resolve().parent;RUN=V7.parent/'reference_condition_v18'
source=(CODE/'evaluate_context_development_v17.py').read_text()
changes={
    "['control','bridge']":"['control','reference']",
    "DATA/variant/'dense_data.pt'":"DATA/'dense_data.pt'",
    "DATA/variant/'risk_dense/risk_probabilities.pt'":"DATA/'risk_dense/risk_probabilities.pt'",
    "outputs['bridge']":"outputs['reference']",
    'bridge_vs_control':'reference_vs_control',
    'Bridge/control isolates addedRGBcontext; control/v16 also includes label-consistency change.':'Reference/control isolates additionalreferencecondition; control/v16 also includes label-consistency change.'}
generated=source
for before,after in changes.items():assert before in generated;generated=generated.replace(before,after)
snapshot=RUN/'evaluator_snapshot.py';snapshot.write_text(generated);save(RUN/'evaluation_provenance.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),replacements=changes))
ns=dict(__name__='reference_eval_v18',__file__=str(snapshot));exec(compile(generated,str(snapshot),'exec'),ns);ns['RUN']=RUN;ns['DATA']=V7.parent/'context_data_v17/control';ns['main']()
