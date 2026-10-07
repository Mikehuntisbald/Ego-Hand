"""Seal two full 3D core predictions, then read real 2D finger labels."""
import json,hashlib,collections
from pathlib import Path
import numpy as np,torch
from online_parameter_model_v47 import OnlineVisual
from instance_parameter_model_v51_r2 import InstanceParameterHand
from complete_instance_v51_r1 import pinhole_project
from cache_instance_conditions_v51 import RUN

def main():
    torch.set_num_threads(4);device='cuda:0';folder=RUN/'surgical_pose';output=RUN/'paired_protocol/localizer_refinement_r1/surgical_evaluation';output.mkdir(exist_ok=True);saved=torch.load(folder/'inputs.pt',weights_only=False,mmap=True)
    inputs=saved['inputs'];meta=saved['metadata'];ids=[i for i,r in enumerate(meta) if r['split']=='dev'];pixels=np.load(folder/'pixels.npy',mmap_mode='r')
    paths=dict(reference=Path('/mnt/why/HOT3D/experiments/online_rgb_iterative_v48/protected/best.pt'),candidate=RUN/'paired_protocol/localizer_refinement_r1/core_r1/dit_joint/best.pt')
    frozen={};checkpoint_info={}
    for name,path in paths.items():
        ck=torch.load(path,weights_only=False,map_location='cpu');model=InstanceParameterHand('dit',device).to(device).eval();missing,unexpected=model.load_state_dict(ck['model'],strict=False)
        assert not unexpected and all(k.startswith(('instance_','ownership_head.')) for k in missing)
        visual=OnlineVisual(device,False);visual.load_tail(ck['visual_tail']);visual.configure();pred=[];visibility=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for begin in range(0,len(ids),4):
                ix=ids[begin:begin+4];b={k:v[ix].to(device) for k,v in inputs.items()};live=visual.encode_pixels([pixels[i] for i in ix],device)
                b['rgb_native']=torch.zeros(len(ix),17,192,1280,device=device);b['rgb_native'][:,8]=live
                p=model.generate(b,2026100753+begin);pred.append(pinhole_project(p['xyz'][:,8],b['camera_params']).float().cpu()/1408);visibility.append(p['encoded'][3]['visibility_probability'][:,8].float().cpu())
        frozen[name]=dict(xy=torch.cat(pred),visibility=torch.cat(visibility));checkpoint_info[name]=dict(path=str(path),sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),step=ck['step'])
        del model,visual;torch.cuda.empty_cache()
    target_path=output/'sealed_dev_predictions.pt';torch.save(dict(predictions=frozen,indices=ids,checkpoints=checkpoint_info),target_path)
    (output/'prediction_freeze.json').write_text(json.dumps(dict(complete=True,sha256=hashlib.file_digest(target_path.open('rb'),'sha256').hexdigest(),GT_2D_not_in_inference=True,GT_3D=False,conditional_human_ROI_diagnostic=True),indent=2))
    # Separate supervision file is opened only after both inference arrays seal.
    gt=torch.load(folder/'targets.pt',weights_only=False);valid=gt['valid'][ids];visible=gt['visible'][ids];xy=gt['xy'][ids]
    scale=(inputs['roi'][ids,8,2:]-inputs['roi'][ids,8,:2]).mean(-1).clamp_min(.01)
    error={k:(v['xy']-xy).norm(dim=-1)/scale[:,None] for k,v in frozen.items()};report={}
    for name,e in error.items():
        report[name]={}
        for label,mask in [('all_labelled',valid),('visible',valid&visible),('annotated_occluded',valid&~visible)]:
            report[name][label]=dict(points=int(mask.sum()),mean_box_normalized_error=float(e[mask].mean()),PCK10=float((e[mask]<=.1).float().mean()))
    good=valid&(error['reference']<=.05);bad=valid&(error['reference']>.1)
    groups=collections.defaultdict(list)
    for i,index in enumerate(ids):groups[meta[index]['group']].append(i)
    differences=[]
    for indices in groups.values():
        mask=valid[indices];differences.append(float((error['candidate'][indices]-error['reference'][indices])[mask].mean()))
    rng=np.random.default_rng(2026100753);means=[np.mean(rng.choice(differences,len(differences),replace=True)) for _ in range(2000)]
    report.update(groups=len(groups),instances=len(ids),correct_points=int(good.sum()),correct_points_harmed_PCK10=int((good&(error['candidate']>.1)).sum()),bad_points=int(bad.sum()),bad_recovered_PCK10=int((bad&(error['candidate']<=.1)).sum()),source_video_paired_CI95=list(map(float,np.quantile(means,[.025,.975]))),GT_3D=False,scope='Conditional pose diagnostic with human boxes; neither full detector performance nor 3D-mm proof')
    (output/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
