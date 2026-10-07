"""RGB temporal parameter samples and label-free whole-segment selection."""
import collections,hashlib,json,time
import numpy as np,torch
from hand3d_v8_common import V7,save
from parameter_codec_v31 import OUT,observation_batch
from semantic_parameter_model_v36 import SemanticParameterHand
from evaluate_joint_kinematic_v30 import subset
from hand3d_rollout_v8 import project_fisheye624


def load_observations(device):
    source=V7.parent/'joint_mano_v28';allrows=json.loads((source/'rows.json').read_text())
    ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select']);rows=[allrows[i] for i in ids]
    cache=subset(torch.load(source/'candidates.pt',weights_only=False,mmap=True),ids,len(allrows),device)
    return rows,ids,allrows,cache


def generate(model_run,device='cuda:0',draws=4,steps=10,seed=202610131,progress=None):
    rows,ids,allrows,cache=load_observations(device);root=V7.parent;source=root/'joint_mano_v28'
    data=torch.load(root/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    infer={k:v for k,v in data.items() if k not in ['gt','valid','gt_uv','uv_valid','original_base_for_evaluation']}
    inputs=torch.load(source/'inputs.pt',weights_only=False,mmap=True)
    infer.update({k:v[ids] for k,v in inputs.items() if k!='center_ids'})
    infer={k:v.to(device) if torch.is_tensor(v) else v for k,v in infer.items()}
    coarse=torch.load(root/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True).to(device)
    right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False).to(device)
    bank=torch.load(root/'side_data_v16/consensus/native_bank.pt',weights_only=False,mmap=True).to(device)
    path=root/model_run;assert json.loads((path/'done.json').read_text())['complete']
    ck=torch.load(path/'best.pt',weights_only=False,map_location=device)
    assert 'coarse_seed' in ck['config'] and 'query_semantics' in ck['config']
    model=SemanticParameterHand(ck['kind'],device).to(device).eval();model.load_state_dict(ck['model'])
    mean=[];values=[];states=[];sides=[];heat=[];start=time.time()
    with torch.inference_mode():
        for begin in range(0,len(rows),8):
            ix=torch.arange(begin,min(begin+8,len(rows)),device=device)
            b=observation_batch(infer,ix,cache['risk'],coarse,right,preserve_fitted=True);b['rgb_native']=bank[infer['feature_ids'][ix]]
            def predict(bb):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    enc=model.encode(bb);delta=model.sample_trajectory(bb,enc,steps,draws,seed+begin)
                    state=bb['kinematic_coarse'][None]+delta;side=enc[3]['side_logits'].argmax(-1)
                    xyz=torch.stack([model.codec.decode(x,side)[:,8] for x in state])
                    average=state.mean(0);meanxyz=model.codec.decode(average,side)[:,8]
                return state.float(),xyz.float(),average.float(),meanxyz.float(),side,enc[3]
            state,xyz,av,avxyz,side,h=predict(b)
            if begin==0:
                poisoned=dict(b,gt=torch.full((len(ix),17,20,3),float('nan'),device=device),valid=torch.zeros(len(ix),20,device=device,dtype=torch.bool),gt_shape=torch.ones(len(ix),10,device=device)*999)
                other=predict(poisoned);assert torch.equal(state,other[0]) and torch.equal(xyz,other[1])
            states.append(state[:,:,8].transpose(0,1).cpu());values.append(xyz.transpose(0,1).cpu())
            mean.append(dict(xyz=avxyz.cpu(),parameter_state=av[:,8].cpu()));sides.append(side.cpu());heat.append(h['xy'][:,8].float().cpu())
            if begin%400==0 and progress:progress(dict(done=min(begin+8,len(rows)),total=len(rows),seconds=time.time()-start))
    return dict(states=torch.cat(states),xyz=torch.cat(values),right=torch.cat(sides),
        mean_observation=dict(xyz=torch.cat([x['xyz'] for x in mean]),parameter_state=torch.cat([x['parameter_state'] for x in mean]),right=torch.cat(sides)),
        heat_xy=torch.cat(heat),kind=ck['kind'],step=ck['step'],draws=draws if ck['kind']=='dit' else 1,ddim_steps=steps,
        seed=seed,batch_size=8,seconds=time.time()-start,gt_poison_exact=True,
        model_run=str(path),checkpoint_sha256=hashlib.sha256((path/'best.pt').read_bytes()).hexdigest())


def huber(value,threshold=.5):
    a=np.abs(value);return np.where(a<threshold,.5*a*a/threshold,a-.5*threshold)


def select_sequence(samples,cache,rows):
    """Second-order Viterbi. State is a draw index, not a GT error rank.

Unary cost uses shared frozen WiLoR/RGB evidence, transition costs use real
world pose/root velocities and accelerations. Different sample indices may
join only as a jointly scored path; final v42 constraints still apply.
"""
    device=cache['base'].device;x=samples['xyz'].to(device);n,k=x.shape[:2]
    camera=(1-cache['risk'][:,:,0]).clamp_min(.03)[:,None]
    pose_weight=(1-cache['risk'][:,:,1]).clamp_min(.03)[:,None]
    base=cache['base'];relative=x-x[:,:,5:6];bp=base-base[:,5:6]
    unary=(camera*torch.tensor(huber(((x-base[:,None])/.03).cpu().numpy()).sum(-1),device=device)
           +pose_weight*torch.tensor(huber(((relative-bp[:,None])/.02).cpu().numpy()).sum(-1),device=device)).mean(-1)
    uv=project_fisheye624(x.flatten(0,1),cache['camera_params'][:,None].expand(-1,k,-1).flatten(0,1)).reshape(n,k,20,2)/1408
    size=(cache['roi'][:,2:]-cache['roi'][:,:2]).mean(-1).clamp_min(.01)
    rgb=torch.tensor(huber(((uv-cache['heat_xy'][:,None])/size[:,None,None,None]*10).cpu().numpy()).sum(-1),device=device)
    unary+=(.15*pose_weight*rgb).mean(-1)
    world=torch.einsum('nsjc,nkc->nsjk',x,cache['rotation'])+cache['translation'][:,None,None]
    poses=(world-world[:,:,5:6]).cpu().numpy();roots=world[:,:,5].cpu().numpy();unary=unary.cpu().numpy().astype(np.float64)
    states=samples['states'].flatten(2).numpy();shape=states[:,:,29:34];side=samples['right'].numpy()
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    chosen=np.empty(n,dtype=np.int64);records=[]
    for key,indices in sorted(groups.items()):
        indices.sort(key=lambda i:rows[i]['timestamp_ns'])
        majority=int(np.mean(side[indices])>=.5)
        # Side inconsistent observations are also downweighted by v42's
        # parameter repair. Don't let their unary cost dictate a path here.
        unary[indices]*=np.where(side[indices]==majority,1.,.05)[:,None]
        segments=[];segment=[]
        for i in indices:
            if segment and (rows[i]['timestamp_ns']-rows[segment[-1]]['timestamp_ns'])/1e9>.55:
                segments.append(segment);segment=[]
            segment.append(i)
        if segment:segments.append(segment)
        for ix in segments:
            count=len(ix);t=np.asarray([rows[i]['timestamp_ns']/1e9 for i in ix]);dt=np.diff(t)
            if count==1:chosen[ix[0]]=int(unary[ix[0]].argmin());continue
            pv=[];rv=[];edge=[]
            for a,b,d in zip(ix,ix[1:],dt):
                p=(poses[b][None]-poses[a][:,None])/d;r=(roots[b][None]-roots[a][:,None])/d
                pn=np.linalg.norm(p,axis=-1)/.65;rn=np.linalg.norm(r,axis=-1)/1.2
                cost=.10*huber(pn).mean(-1)+.04*huber(rn)
                cost+=.25*huber(np.maximum(pn-1,0)).mean(-1)+.25*huber(np.maximum(rn-1,0))
                cost+=.01*((shape[b][None]-shape[a][:,None])**2).mean(-1)
                pv.append(p);rv.append(r);edge.append(cost)
            dynamic=unary[ix[0]][:,None]+unary[ix[1]][None]+edge[0];back={}
            for j in range(2,count):
                h=(dt[j-2]+dt[j-1])/2
                ap=(pv[j-1][None]-pv[j-2][:,:,None])/h
                ar=(rv[j-1][None]-rv[j-2][:,:,None])/h
                penalty=.05*huber(np.linalg.norm(ap,axis=-1)/10).mean(-1)+.02*huber(np.linalg.norm(ar,axis=-1)/12)
                total=dynamic[:,:,None]+edge[j-1][None]+unary[ix[j]][None,None]+penalty
                back[j]=total.argmin(0);dynamic=total.min(0)
            previous,last=np.unravel_index(dynamic.argmin(),dynamic.shape);path=[0]*count;path[-2:]=[int(previous),int(last)]
            for j in range(count-1,1,-1):path[j-2]=int(back[j][path[j-1],path[j]])
            chosen[ix]=path;records.append(dict(track=list(key),frames=count,sample_index_changes=int(np.count_nonzero(np.diff(path))),cost=float(dynamic.min())))
    rows_index=torch.arange(n);sample_index=torch.tensor(chosen)
    obs=dict(xyz=samples['xyz'][rows_index,sample_index],parameter_state=samples['states'][rows_index,sample_index],right=samples['right'])
    return obs,dict(method='Second-order whole-segment Viterbi with common observation/RGB evidence and world speed/acceleration',
                    uses_gt=False,selected_index=chosen.tolist(),segments=records,
                    candidate_counts=np.bincount(chosen,minlength=k).tolist(),independent_frame_argmin=False)
