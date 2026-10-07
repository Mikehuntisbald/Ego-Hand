"""As-is inference audit: original pixels and original observation flags."""
import json
import spatial_rgb_common as s
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_spatial_temporal import load
from offline_kp_model import condition
from spatial_temporal_model import SpatialTemporalCompleter
from spatial_rgb_model import SpatialHead
from offline_rgb_encoder import crop_image

torch.set_num_threads(4);O=s.RUN/'natural_finger_audit';A=O/'performance';A.mkdir(exist_ok=True)
rows,data=load('cuda:0');candidates=json.loads((O/'candidates.json').read_text());selected=json.loads((O/'reviewed_examples.json').read_text())
ids=torch.tensor([i for i,r in enumerate(rows) if r['role']=='test'],device='cuda:0');lookup={wi:j for j,wi in enumerate(ids.tolist())}
pred={};probe=SpatialHead().to('cuda:0').eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location='cuda:0')['model'])
with torch.inference_mode():
    for arm in ['rgb_dit','rgb_regression']:
        ck=torch.load(s.RUN/'sealed'/f'{arm}.pt',weights_only=False,map_location='cuda:0');model=SpatialTemporalCompleter(ck['kind'],True).to('cuda:0').eval();model.load_state_dict(ck['model'])
        out=[];rgb=[]
        for start in range(0,len(ids),48):
            ix=ids[start:start+48];fids=data['feature_ids'][ix]
            b=condition(data['xy'][ix],data['observed'][ix],data['dt'][ix])
            b.update(rgb=data['bank'][fids,0],positions=data['positions_bank'][fids],roi=data['roi'][fids],rgb_valid=fids>0)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p=model.predict(b)['xy']
                if arm=='rgb_dit':
                    x=b['rgb'][:,8].float().transpose(1,2).reshape(-1,128,16,12)
                    q=probe.decode(x,b['positions'][:,8],b['roi'][:,8])['xy'];rgb.append(q.cpu().numpy())
            assert torch.equal(p[b['observed'][:,8]],b['xy'][:,8][b['observed'][:,8]])
            out.append(p.cpu().numpy())
        pred[arm]=np.concatenate(out)
        if rgb:pred['rgb_only_probe']=np.concatenate(rgb)
pred['wilor_input']=data['xy'][ids,8].cpu().numpy()
gt=data['gt'][ids].cpu().numpy();valid=data['valid'][ids].cpu().numpy();valid[:,5]=False
observed=data['observed'][ids,8].cpu().numpy()
def metrics(js):
    m=valid[js];common=m&observed[js];result={}
    for arm,p in pred.items():
        e=np.linalg.norm(p[js]-gt[js],axis=-1)*1408
        use=common if arm=='wilor_input' else m
        result[arm]=dict(mean_px=float(e[use].mean()),pck20=float((e[use]<=20).mean()),evaluated_points=int(use.sum()),
            common_observed_mean_px=float(e[common].mean()),tips_mean_px=float(e[:,:5][use[:,:5]].mean()),p95_px=float(np.quantile(e[use],.95)))
    result['status']=dict(windows=len(js),gt_valid_finger_points=int(m.sum()),input_observed_valid_points=int(common.sum()),
        genuinely_unprovided_valid_points=int((m&~observed[js]).sum()),all_20_marked_observed_windows=int(observed[js].all(1).sum()),
        corrected_observed_points=int(((np.linalg.norm(pred['rgb_dit'][js]-pred['wilor_input'][js],axis=-1)>1e-7)&common).sum()))
    return result
result=dict(scope='Original RGB, original observation flags. No synthetic pixel occlusion or coordinate deletion. All valid finger joints evaluated, not an invisible-only metric.',
    rgb_probe_scope='Standalone frozen visual localization head for diagnosis; not the deployed DiT output',
    all_test=metrics(np.arange(len(ids))),candidates=metrics(np.array([lookup[q['window_index']] for q in candidates])),examples=[])
for q in selected:
    j=lookup[q['window_index']];result['examples'].append(dict(id=q['id'],category=q['visual_review_category'],sequence=q['sequence'],clip=q['clip'],frame=q['frame'],**metrics(np.array([j]))))
s.save(A/'results.json',result)
np.savez_compressed(A/'predictions.npz',window_indices=ids.cpu().numpy(),gt=gt,valid=valid,observed=observed,**pred)
records,index=s.records_and_index()
for page,start in enumerate(range(0,len(selected),4)):
    fig,axs=plt.subplots(4,3,figsize=(10,12),layout='constrained')
    for rr,q in enumerate(selected[start:start+4]):
        wi=q['window_index'];j=lookup[wi];fid=int(index['feature_ids'][wi,8]);roi=index['roi'][fid].numpy()*1408
        image=np.rot90(crop_image(cv2.imread(q['image']),roi)[:,:,::-1])
        for cc,arm in enumerate([None,'rgb_dit','rgb_only_probe']):
            ax=axs[rr,cc];ax.imshow(image);ax.axis('off');ax.set_xlim(0,255);ax.set_ylim(255,0)
            if arm is None:ax.set_title(f"ID {q['id']} | {q['visual_review_category']}\nOriginal RGB",fontsize=9);continue
            for pts,col,mrk in [(gt[j],'#00f5c4','o'),(pred[arm][j],'#ff5252','x')]:
                uv=(pts*1408-roi[:2])/(roi[2:]-roi[:2])*256;uv=np.stack([uv[:,1],255-uv[:,0]],-1)
                ax.scatter(uv[valid[j],0],uv[valid[j],1],s=14,c=col,marker=mrk,linewidths=1)
            e=np.linalg.norm(pred[arm][j]-gt[j],axis=-1)[valid[j]].mean()*1408
            ax.set_title(('As-is DiT (preserves input)' if arm=='rgb_dit' else 'RGB-only diagnostic head')+f'\nMean finger error {e:.1f} px',fontsize=9)
    fig.suptitle('Original natural cases | Reference green, prediction red | No artificial missing input',fontsize=11)
    fig.savefig(A/f'overlay_{page}.png',dpi=140);plt.close(fig)
print(json.dumps(result,indent=2))
