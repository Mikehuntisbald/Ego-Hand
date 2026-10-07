"""Remove all historical conditioning, including the temporal error-risk features."""
import json
import numpy as np,torch
from hand3d_v8_common import RUN,V7,load,batch,metrics,save
from hand3d_risk_v7 import Risk3D
from hand3d_data_v7 import risk_features
from evaluate_hand3d_v8 import restore
from bounded_policy_v8 import apply

@torch.no_grad()
def main():
    torch.set_num_threads(4);device='cuda:0';data=load(device);ids=torch.tensor(np.where(np.asarray(data['roles'])=='dev_calibrate')[0],device=device);selection=json.loads((RUN/'development_selection.json').read_text())['selected'];folder=selection['folder'];policy=selection['policy'];model,visual,ck=restore(folder,device,data)
    rck=torch.load(V7/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(rck['dim']).to(device).eval();risk.load_state_dict(rck['model']);temp=torch.tensor(json.loads((V7/'risk_calibration.json').read_text())['temperature'],device=device);outputs={}
    for mode in ['full','no_history']:
        parts=[]
        for start in range(0,len(ids),16):
            ix=ids[start:start+16];b=batch(data,ix)
            if mode=='no_history':
                keep=torch.zeros(len(ix),17,device=device,dtype=torch.bool);keep[:,8]=True
                for k in ['xyz','xy','rgb','risk_rgb','positions','camera_origin','roi']:
                    value=b[k];mask=keep.reshape(*keep.shape,*([1]*(value.ndim-2)));b[k]=value*mask
                b['available']&=keep[...,None];b['observed_2d']&=keep[...,None];b['rgb_valid']&=keep;b['scores']*=keep
            pr=(risk(risk_features(b))/temp).sigmoid();b['risk_camera']=pr[:,:,0];b['risk_relative']=pr[:,:,1]
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model.predict(b,seed=202610081+start)
            parts.append(pred[policy['source']].float())
        base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];pred=apply(torch.cat(parts),base,policy);outputs[mode]=metrics(pred,base,data['gt'][ids],data['valid'][ids])
    save(RUN/'history_diagnosis.json',dict(results=outputs,scope='Development perturbation only; current camera/RGB retained, historical RGB/XYZ/metadata and temporal risk features erased; no occlusion-recovery claim'));print(json.dumps(outputs),flush=True)

if __name__=='__main__':main()
