import json,torch
from hand3d_v8_common import V7,save,metrics
from hand3d_data_v7 import batch
from density_model_v13 import DensityTrajectoryHand3D
from reference_condition_v18 import ReferenceConditionHand3D
from train_aligned_density_v13 import make
from temporal_reference_v14 import reference
RUN=V7.parent/'reference_condition_v18';DATA=V7.parent/'context_data_v17/control'

def main():
    torch.set_num_threads(4);device='cuda:3';RUN.mkdir(exist_ok=True)
    raw=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()};bank=torch.load(DATA/'native_bank.pt',weights_only=False,mmap=True).to(device);risk=torch.load(DATA/'risk_dense/risk_probabilities.pt',weights_only=False)['joint'].to(device)
    ids=torch.tensor([i for i,r in enumerate(data['roles']) if r=='dev_select'],device=device);b=make(data,bank,ids[:2],risk);policy=json.loads((V7.parent/'side_native_v16/fifth_seal.json').read_text())['policy'];ck=torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt',weights_only=False,map_location=device)
    baseline=DensityTrajectoryHand3D('dit',True).to(device).eval();baseline.load_state_dict(ck['model']);model=ReferenceConditionHand3D(policy).to(device).eval();mismatch=model.load_state_dict(ck['model'],strict=False);assert all(k.startswith('reference_project.') for k in mismatch.missing_keys) and not mismatch.unexpected_keys
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=baseline.predict_rollout(b,seed=118)['xyz_camera_m'];p=model.predict_rollout(b,seed=118)['xyz_camera_m'];assert torch.equal(a,p)
        poisoned={**b,'gt':torch.ones_like(b['base'])*999,'gt_side':'invented','visibility':torch.zeros_like(b['available'])};q=model.predict_rollout(poisoned,seed=118)['xyz_camera_m'];assert torch.equal(p,q)
    extra={k:v.to(device) for k,v in torch.load(DATA/'trajectory_labels.pt',weights_only=False).items()};model.train();model.localization.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.rollout_loss(b,*[extra[k][ids[:2]] for k in ['gt','valid','gt_uv','uv_valid']],118)
    assert torch.isfinite(loss);loss.backward();grad=float(model.reference_project[-1].weight.grad.norm());assert grad>0
    result=dict(passed=True,initial_sampler_exact_parity=True,GT_fields_exact_invariance=True,output_shape=list(p.shape),reference_output_gradient=grad,loss=float(loss),base_coordinates_unchanged=True,scope='Engineeringpreflight; no recovery claim; reference from observedXYZ/scores/times only, all spatialRGB retained')
    save(RUN/'preflight.json',result)
    with torch.no_grad():
        raw_reference,usable=reference(batch(data,ids),horizon=.5,robust=True);base=data['original_base_for_evaluation'][ids];m=metrics(raw_reference,base,data['gt'][ids],data['valid'][ids]);save(RUN/'reference_diagnostic.json',dict(dev_select=m,usable_windows=int(usable.sum()),scope='GTfree fit, labels only score afterward; diagnostic not deployable correction'))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
