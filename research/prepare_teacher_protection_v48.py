"""Frozen admitted teacher on natural RGB; labels stay outside inference."""
import hashlib,json,time
import numpy as np,torch
from hand3d_v8_common import V7,save
from online_parameter_model_v47 import RUN as SOURCE,OnlineVisual,load_head
from parameter_codec_v31 import OUT,observation_batch

RUN=V7.parent/'online_rgb_iterative_v48'
def main():
    torch.set_num_threads(4);device='cuda:3';RUN.mkdir(exist_ok=True);start=time.time()
    if (RUN/'teacher_ready.json').exists():return
    data=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in data.items()}
    target=torch.load(SOURCE/'targets.pt',weights_only=False,mmap=True)
    coarse=torch.load(V7.parent/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True).to(device)
    right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False).to(device)
    risks=torch.load(SOURCE/'risk.pt',weights_only=False);prob=risks['train_oof'].to(device)
    ckpath=SOURCE/'dit_joint/best.pt';ck=torch.load(ckpath,weights_only=False,map_location=device)
    model,_,_=load_head('dit',device);model.load_state_dict(ck['model']);model.eval()
    visual=OnlineVisual(device,False);visual.load_tail(ck['visual_tail']);visual.configure();pixels=np.load(SOURCE/'pixels.npy',mmap_mode='r')
    selected=torch.tensor([i for i,r in enumerate(data['roles']) if r in ['train','dev_select']],device=device)
    fids=torch.unique(data['feature_ids'][selected]);fids=fids[fids>0]
    teacher=torch.zeros(len(data['roles']),20,3)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        bank=torch.zeros(len(right),192,1280,device=device)
        for begin in range(0,len(fids),128):
            ix=fids[begin:begin+128];bank[ix]=visual.encode_pixels([pixels[i] for i in ix.cpu().tolist()],device)
            if begin%2048==0:print(json.dumps(dict(stage='teacher_RGB',done=min(begin+128,len(fids)),total=len(fids),seconds=time.time()-start)),flush=True)
        for begin in range(0,len(selected),8):
            ix=selected[begin:begin+8];b=observation_batch(data,ix,prob,coarse,right,preserve_fitted=True);b['rgb_native']=bank[data['feature_ids'][ix]]
            teacher[ix.cpu()]=model.predict_parameters(b,seed=2026100517+begin)['xyz_camera_m'].cpu()
    torch.save(teacher,RUN/'teacher_centers.pt')
    gt=target['gt'][:,8];valid=target['valid'][:,8].clone();valid[:,5]=False
    train=torch.tensor([r=='train' for r in data['roles']]);mask=valid&train[:,None]
    raw=data['xyz_camera_bank'][data['feature_ids'][:,8]].cpu()
    err=lambda p:(p-gt).norm(dim=-1);rel=lambda p:((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)
    evidence={}
    for name,fn in [('camera',err),('relative',rel)]:
        original=mask&(fn(raw)<=.01);good=mask&(fn(teacher)<=.01)
        evidence[name]=dict(original_WiLoR_good_points=int(original.sum()),teacher_good_points=int(good.sum()),
            teacher_good_not_covered_by_original_guard=int((good&~original).sum()))
    save(RUN/'teacher_ready.json',dict(complete=True,checkpoint=str(ckpath),checkpoint_step=ck['step'],
        checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),teacher_sha256=hashlib.sha256((RUN/'teacher_centers.pt').read_bytes()).hexdigest(),
        GT_used_for_training_loss_masks_only=True,teacher_not_an_inference_input=True,protected_error_threshold_m=.01,
        natural_RGB_and_online_BF16_padded16=True,train_mask_evidence=evidence,seconds=time.time()-start))
    print((RUN/'teacher_ready.json').read_text(),flush=True)
if __name__=='__main__':main()
