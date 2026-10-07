"""Final evidence checks, including actual visual updates versus warm600."""
import hashlib,json
import torch
from hand3d_v8_common import V7,save
from online_parameter_model_v47 import RUN as SOURCE
RUN=V7.parent/'online_rgb_iterative_v48'
torch.set_num_threads(4)
summary=json.loads((RUN/'summary.json').read_text());assert summary['complete']
hashes=json.loads((RUN/'code_hashes.json').read_text())
for name,value in hashes.items():assert hashlib.sha256((RUN/'code_snapshot'/name).read_bytes()).hexdigest()==value
evidence={}
for arm in ['frozen','joint','protected']:
    folder=RUN/arm;done=json.loads((folder/'done.json').read_text());assert done['complete'] and done['steps']==1800
    preflight=json.loads((folder/'preflight.json').read_text());resume=json.loads((folder/'resume_verification.json').read_text())
    assert preflight['passed'] and resume['passed']
    source=SOURCE/('dit_frozen' if arm=='frozen' else 'dit_joint')/'last.pt'
    warm=torch.load(source,weights_only=False,map_location='cpu');last=torch.load(folder/'last.pt',weights_only=False,map_location='cpu')
    assert last['step']==1800 and last['cumulative_step']==2400
    head=float((last['model']['semantic_head.weight']-warm['model']['semantic_head.weight']).abs().max());assert head>0
    for key,value in warm['model'].items():
        if key.startswith('codec.'):assert torch.equal(value,last['model'][key]),key
    changes={}
    if arm=='frozen':assert last['visual_tail'] is None
    else:
        for prefix in ['blocks.28.','blocks.29.','blocks.30.','blocks.31.','last_norm.']:
            changes[prefix]=max(float((v-warm['visual_tail'][k]).abs().max()) for k,v in last['visual_tail'].items() if k.startswith(prefix))
        assert all(v>0 for v in changes.values()),changes
    selected=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')
    if selected['config'].get('reference_only',False):
        teacher=torch.load(SOURCE/'dit_joint/best.pt',weights_only=False,map_location='cpu')
        assert all(torch.equal(v,teacher['model'][k]) for k,v in selected['model'].items())
    else:
        entries=json.loads((folder/'history.json').read_text());entry=[x for x in entries if x['step']==selected['step']][0]
        assert entry['feasible']
    evidence[arm]=dict(semantic_head_max_update=head,visual_max_updates=changes,FK_geometry_buffers_exact=True,
        selected_reference_only=selected['config'].get('reference_only',False),resume_passed=True)
    del last,warm,selected
for mode,seal in summary['seals'].items():
    assert seal['generator_gt_poison_exact'] and seal['selector_gt_poison_exact'] and seal['parameter_redecode_passed'] and seal['check']['passed']
    path=RUN/'diagnostic_v46'/mode/'result.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['result_sha256']
result=dict(complete=True,passed=True,actual_weight_updates=evidence,sealed_GT_free_inference_and_redecode_passed=True,
    frozen_code_snapshot_hashes_exact=True,default_changed=False,scope='Engineering and integrity checks, not performance admission or fresh generalization')
save(RUN/'verification.json',result);save(RUN/'review/verification.json',result)
print(json.dumps(result),flush=True)
