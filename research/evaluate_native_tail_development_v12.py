import json,torch
from hand3d_v8_common import V7,load,save,metrics
from native_projection_policy_v11 import apply
from calibrate_hand3d_v8 import paired_ci
RUN=V7.parent/'native_tail_v12'

def main():
    torch.set_num_threads(4);data=load('cpu');policy=dict(strength=1.,cap_m=.0095,root_cap_m=.0095,relative_weight=4.)
    results={};pred={};raw={}
    for mode in ['frozen','ft']:
        assert json.loads((RUN/mode/'done.json').read_text())['complete'];c=torch.load(RUN/mode/'calibration.pt',weights_only=False);ids=c['indices'];base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];pred[mode]=apply(c['proposal'],base,policy);raw[mode]=c['proposal']
        results[mode]=dict(final=metrics(pred[mode],base,gt,valid),raw=metrics(raw[mode],base,gt,valid),selected_step=json.loads((RUN/mode/'done.json').read_text())['selected_step'])
    ci=paired_ci(pred['ft'],pred['frozen'],gt,valid,rows);native=torch.load(V7.parent/'native_critic_v10/critic_calibration.pt',weights_only=False);assert torch.equal(ids,native['indices'].cpu());current=apply(native['proposal'],base,policy)
    current_metrics=metrics(current,base,gt,valid);qualified=results['ft']['final']['relative_mm']<results['frozen']['final']['relative_mm']-.1 and results['ft']['final']['relative_mm']<current_metrics['relative_mm']-.1 and ci['relative']['ci95_delta_mm'][1]<0
    out=dict(completed=True,results=results,ft_minus_frozen_ci=ci,current_v11=current_metrics,backbone_finetune_warrants_fresh_validation=qualified,scope='Development only; third12clips already evaluated, cannot be fresh forv12; no new independent metric opened',next='Freeze new unused clips only if FT clears paired development gain; otherwise retain failure and improve dense natural observations')
    save(RUN/'development_results.json',out);print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
