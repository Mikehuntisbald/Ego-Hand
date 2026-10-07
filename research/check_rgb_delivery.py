import json
import wilor_eval_common
import numpy as np,torch
from offline_rgb_data import RUN,save
from offline_rgb_model import RGBKeypointCompleter
from offline_rgb_adapter import RGBAdapterCompleter
from train_offline_rgb import load,batch

torch.set_num_threads(4);rows,data=load('cuda:2')
ids=torch.tensor([i for i,r in enumerate(rows) if r['role']=='development'][:8],device='cuda:2')
b,_=batch(data,ids,torch.full((8,),6,device='cuda:2'),1+ids%4)
checks={}
for arm in ['rgb_dit','rgb_regression']:
    ck=torch.load(RUN/'sealed'/f'{arm}.pt',map_location='cuda:2',weights_only=False)
    cls=RGBAdapterCompleter if ck.get('architecture')=='adapter' else RGBKeypointCompleter
    model=cls(ck['kind'],True).to('cuda:2').eval();model.load_state_dict(ck['model'])
    swapped=dict(b);swapped['rgb']=torch.roll(b['rgb'],1,0)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict(b)['xy'];q=model.predict(swapped)['xy']
    difference=float((p-q).norm(dim=-1).mean()*1408)
    assert difference>1e-4,'RGB pixels do not affect the delivered prediction'
    known=dict(b);known['xy']=b['xy'].clone();known['observed']=b['observed'].clone();known['missing']=b['missing'].clone()
    known['xy'][:,8,5]=data['xy'][ids,8,5];known['observed'][:,8,5]=True;known['missing'][:,5]=False
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model.predict(known)['xy']
    assert torch.equal(out[:,5],known['xy'][:,8,5])
    checks[arm]=dict(rgb_swap_changes_output_mean_px=difference,provided_point_preserved_exactly=True,
        visual_gain=float(model.visual_gain.detach()) if hasattr(model,'visual_gain') else None,
        local_gain=float(model.local_gain.detach()) if hasattr(model,'local_gain') else None)
inp=json.loads((RUN/'delivery/example_input.json').read_text())
out=json.loads((RUN/'delivery/example_rgb_dit.json').read_text());count=0
for f,p in zip(inp['tracks'][0]['frames'],out['tracks'][0]['frames']):
    for j,seen in enumerate(f['observed']):
        if seen:assert p['points'][j]['xy_px']==f['xy_px'][j];count+=1
        assert p['points'][j]['review_required'] and not p['points'][j]['reviewed']
save(RUN/'delivery_checks.json',dict(models=checks,rgb_image_to_annotation_cli_executed=True,
    input_coordinates_preserved_in_json=count,all_automatic_points_require_review=True))
print(json.dumps(checks,indent=2))
