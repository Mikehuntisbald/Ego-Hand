import json,argparse
import numpy as np,torch
from hand3d_v8_common import V7,load,batch,save,metrics
from hand3d_native_v10 import NativeTrajectoryHand3D
from hand3d_visual_v8 import dense_bank
from hand3d_risk_v7 import Risk3D
from hand3d_data_v7 import risk_features
from bounded_policy_v8 import apply
from calibrate_hand3d_v8 import paired_ci
RUN=V7.parent/'offline_hand3d_v10_rollout'

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:3');a=p.parse_args();torch.set_num_threads(4);device=a.device;assert (RUN/'rgb_dit/training_done.json').exists()
    data=load(device);bank=dense_bank(device,len(data['world']));ids=torch.tensor(np.where(np.asarray(data['roles'])=='dev_calibrate')[0],device=device);ck=torch.load(RUN/'rgb_dit/best.pt',weights_only=False,map_location=device);model=NativeTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(ck['model']);rck=torch.load(V7/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(rck['dim']).to(device).eval();risk.load_state_dict(rck['model']);temperature=torch.tensor(json.loads((V7/'risk_calibration.json').read_text())['temperature'],device=device)
    gt=data['gt'][ids];valid=data['valid'][ids];base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];rows=[data['rows'][int(i)] for i in ids];reports={};predictions={}
    for mode in ['full','no_history','no_rgb','shuffled_rgb']:
        parts=[]
        for start in range(0,len(ids),8):
            ix=ids[start:start+8];b=batch(data,ix);b['rgb_native']=bank[data['feature_ids'][ix]]
            if mode=='no_history':
                keep=torch.zeros_like(b['rgb_valid']);keep[:,8]=True
                for key in ['xyz','xy','rgb','rgb_native','positions','roi','camera_origin','rays','risk_rgb']:
                    x=b[key];b[key]=x*keep.reshape(*keep.shape,*([1]*(x.ndim-2)))
                b['available']&=keep[...,None];b['observed_2d']&=keep[...,None];b['rgb_valid']&=keep;b['scores']*=keep
            elif mode=='no_rgb':
                for key in ['rgb','rgb_native','risk_rgb']:b[key]=torch.zeros_like(b[key])
            elif mode=='shuffled_rgb':
                for key in ['rgb','rgb_native','risk_rgb']:b[key]=b[key].flip(0)
            pr=(risk(risk_features(b))/temperature).sigmoid();b['risk_camera']=pr[:,:,0];b['risk_relative']=pr[:,:,1]
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model.predict_rollout(b,seed=202610103+start)
            parts.append(out['xyz_camera_m'].float())
        raw=torch.cat(parts);pred=apply(raw,base,dict(cap_m=.0095,strength=.5));predictions[mode]=(raw,pred);reports[mode]=dict(raw=metrics(raw,base,gt,valid),bounded=metrics(pred,base,gt,valid))
    intervals={}
    for mode in ['no_history','no_rgb','shuffled_rgb']:intervals[mode]=dict(raw=paired_ci(predictions[mode][0],predictions['full'][0],gt,valid,rows),bounded=paired_ci(predictions[mode][1],predictions['full'][1],gt,valid,rows))
    save(RUN/'conditioning_diagnosis.json',dict(results=reports,ablations_minus_full_ci=intervals,scope='Development diagnostic; historical metadata and risk cues erased for no_history; RGB cues also erased from risk for no_rgb; not an occlusion benchmark'))
    print(json.dumps({mode:dict(raw_camera=v['raw']['camera_mm'],raw_relative=v['raw']['relative_mm'],bounded_camera=v['bounded']['camera_mm'],bounded_relative=v['bounded']['relative_mm']) for mode,v in reports.items()}),flush=True)

if __name__=='__main__':main()
