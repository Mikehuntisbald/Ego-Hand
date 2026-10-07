"""Trained-head inference checks; GT poison and controlled input perturbations."""
import json,torch
from hand3d_v8_common import V7,save
from parameter_codec_v31 import OUT,observation_batch
from parameter_temporal_model_v31 import ParameterTemporalHand
from semantic_parameter_model_v36 import SemanticParameterHand


def main():
    torch.set_num_threads(4);device='cuda:0'
    data=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    data={k:v.to(device) if torch.is_tensor(v) else v for k,v in data.items()}
    bank=torch.load(V7.parent/'side_data_v16/consensus/native_bank.pt',weights_only=False,mmap=True).to(device)
    right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False).to(device)
    probability=torch.load(V7.parent/'side_data_v16/consensus/risk_dense/risk_probabilities.pt',weights_only=False)['joint'].to(device)
    ids=torch.tensor([i for i,r in enumerate(data['roles']) if r=='dev_select'][:8],device=device)
    results=[]
    for name in ['parameter_kinematic_v31/regression','parameter_kinematic_v31/dit','semantic_parameter_v36/regression','fitted_parameter_v39/regression']:
        run=V7.parent/name;ck=torch.load(run/'best.pt',weights_only=False,map_location=device)
        fitted='coarse_seed' in ck['config'];coarsepath=V7.parent/'observation_ik_v38/coarse_state_bank.pt' if fitted else OUT/'coarse_state_bank.pt'
        coarse=torch.load(coarsepath,weights_only=False,mmap=True).to(device)
        cls=SemanticParameterHand if 'query_semantics' in ck['config'] else ParameterTemporalHand
        model=cls(ck['kind'],device).to(device).eval();model.load_state_dict(ck['model'])
        b=observation_batch(data,ids,probability,coarse,right,preserve_fitted=fitted);b['rgb_native']=bank[data['feature_ids'][ids]]
        def infer(bb):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):return model.predict_parameters(bb,seed=202610131)
        clean=infer(b);poison=dict(b,gt=torch.randn(8,17,20,3,device=device)*999,gt_shape=torch.randn(8,10,device=device),gt_handedness=1-right[data['feature_ids'][ids,8]],valid=torch.zeros(8,20,device=device,dtype=torch.bool),visibility=torch.rand(8,20,device=device),gt_uv=torch.randn(8,20,2,device=device))
        changed=infer(poison)
        assert torch.equal(clean['state'],changed['state']);assert torch.equal(clean['xyz_camera_m'],changed['xyz_camera_m'])
        assert clean['xyz_camera_m'].shape==(8,20,3);assert torch.isfinite(clean['xyz_camera_m']).all()
        no_rgb=infer(dict(b,rgb_native=torch.zeros_like(b['rgb_native'])))
        current=dict(b)
        for key in ['xyz','available','scores','rgb_native','rgb_valid']:
            z=b[key].clone();z[:,:8]=0;z[:,9:]=0;current[key]=z
        now=infer(current)
        result=dict(run=name,selected_step=ck['step'],passed=True,gt_poison_exact=True,output_shape=[8,20,3],
                    no_rgb_output_change_mean_mm=float((no_rgb['xyz_camera_m']-clean['xyz_camera_m']).norm(dim=-1).mean()*1000),
                    current_only_output_change_mean_mm=float((now['xyz_camera_m']-clean['xyz_camera_m']).norm(dim=-1).mean()*1000),
                    scope='8developmentinputs, output sensitivity only; not retraining ablation, visibility truth or accuracy benefit.')
        results.append(result);print(json.dumps(result),flush=True);del model,coarse
    out=V7.parent/'structure_temporal_review_v40';out.mkdir(exist_ok=True);save(out/'trained_inference_checks.json',dict(complete=True,results=results))


if __name__=='__main__':main()
