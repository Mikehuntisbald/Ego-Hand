import json,torch
from hand3d_v8_common import V7,save,metrics
from native_projection_policy_v11 import apply
from density_model_v13 import POLICY
from calibrate_hand3d_v8 import paired_ci
RUN=V7.parent/'anchor_native_v14_closed';DATA=V7.parent/'aligned_density_v13'

def main():
    torch.set_num_threads(4);data=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True);outputs={};reports={}
    for blend in [0.,.5,1.]:
        path=RUN/f'blend{blend:g}';done=json.loads((path/'done.json').read_text());assert done['selected_step']>1200;c=torch.load(path/'calibration.pt',weights_only=False);ids=c['indices'];base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];pred=apply(c['proposal'],base,POLICY);reports[str(blend)]=dict(selected_step=done['selected_step'],raw=metrics(c['proposal'],base,gt,valid),bounded=metrics(pred,base,gt,valid));outputs[str(blend)]=dict(prediction=pred,proposal=c['proposal'])
    comparisons={}
    for blend in [.5,1.]:comparisons[str(blend)]=dict(raw=paired_ci(outputs[str(blend)]['proposal'],outputs['0.0']['proposal'],gt,valid,rows),bounded=paired_ci(outputs[str(blend)]['prediction'],outputs['0.0']['prediction'],gt,valid,rows))
    obj=dict(complete=True,results=reports,anchor_minus_noanchor_ci=comparisons,scope='Matched training/center/nativeRGB/schedule. Only complete-sampler-trained checkpoints eligible. Development only, fourth batch metrics still unopened.')
    save(RUN/'development_results.json',obj);print(json.dumps(obj,indent=2))

if __name__=='__main__':main()
