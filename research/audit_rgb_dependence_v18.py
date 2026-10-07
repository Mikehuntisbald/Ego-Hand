"""Development-only visual evidence perturbation; keep risk and XYZ fixed."""
import json,torch
from hand3d_v8_common import V7,save,metrics
from hand3d_data_v7 import batch
from density_model_v13 import DensityTrajectoryHand3D
from adaptive_projection_v14 import apply
from calibrate_hand3d_v8 import paired_ci
RUN=V7.parent/'reference_condition_v18';DATA=V7.parent/'side_data_v16/consensus'

@torch.inference_mode()
def main():
    torch.set_num_threads(4);device='cuda:3';raw=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()};bank=torch.load(DATA/'native_bank.pt',weights_only=False,mmap=True).to(device);prob=torch.load(DATA/'risk_dense/risk_probabilities.pt',weights_only=False)['joint'].to(device)
    ids=torch.tensor([i for i,r in enumerate(data['roles']) if r=='dev_select'],device=device);ck=torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt',weights_only=False,map_location=device);model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(ck['model']);policy=json.loads((V7.parent/'side_native_v16/fifth_seal.json').read_text())['policy'];base=data['original_base_for_evaluation'][ids];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];out={};reports={}
    for mode in ['original','zero_rgb','time_reverse_rgb','zero_context_rgb']:
        predictions=[];raws=[]
        for start in range(0,len(ids),8):
            ix=ids[start:start+8];b=batch(data,ix,prob);b['rgb_native']=bank[data['feature_ids'][ix]].clone()
            if mode=='zero_rgb':b['rgb_native'].zero_()
            elif mode=='time_reverse_rgb':b['rgb_native']=b['rgb_native'].flip(1)
            elif mode=='zero_context_rgb':
                current=b['rgb_native'][:,8].clone();b['rgb_native'].zero_();b['rgb_native'][:,8]=current
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(b,seed=202610114+start)
            raw_xyz=p['xyz_camera_m'].float();raws.append(raw_xyz);predictions.append(apply(raw_xyz,b,policy))
        raw_xyz=torch.cat(raws);pred=torch.cat(predictions);out[mode]=pred
        reports[mode]=dict(final=metrics(pred,base,gt,valid),raw=metrics(raw_xyz,base,gt,valid))
        if mode!='original':reports[mode]['paired_delta_vs_original']=paired_ci(pred,out['original'],gt,valid,rows)
    save(RUN/'rgb_dependence_diagnostic.json',dict(complete=True,results=reports,scope='Dev_select perturbation only. XYZ,risks,spatialpositions,timestamps and seed fixed; fullnativeRGB drives both semantic/localization stems. Zero/reversal are distribution shifts, not independent training ablations or natural-occlusion benchmarks. No model selection/test metrics used.'))
    print(json.dumps({k:{'camera':v['final']['camera_mm'],'relative':v['final']['relative_mm'],'raw_relative':v['raw']['relative_mm']} for k,v in reports.items()},indent=2),flush=True)
if __name__=='__main__':main()
