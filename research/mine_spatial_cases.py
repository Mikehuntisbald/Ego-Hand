"""Post-hoc error audit only: no training, checkpoint selection, or inference GT."""
import json
from pathlib import Path
import spatial_rgb_common as s
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from offline_rgb_encoder import crop_image

OUT=s.RUN/'hard_case_audit';OUT.mkdir(exist_ok=True)
z=np.load(s.RUN/'rgb_dit_predictions.npz');reg=np.load(s.RUN/'rgb_regression_predictions.npz');base=np.load(s.RUN/'tracks_dit_predictions.npz')
rows=json.loads((s.OLD/'rows.json').read_text());records,index=s.records_and_index();w=torch.load(s.OLD/'windows.pt',weights_only=False)
gtf,vf=s.targets(records,index,None);gtf=gtf.numpy();vf=vf.numpy()
gt=z['gt'];mask=z['mask'];error=np.linalg.norm(z['predicted']-gt,axis=-1)*1408
be=np.linalg.norm(base['predicted']-gt,axis=-1)*1408;re=np.linalg.norm(reg['predicted']-gt,axis=-1)*1408
count=mask.sum(1);avg=lambda x:(x*mask).sum(1)/count.clip(1)
e=avg(error);b=avg(be);r=avg(re);linear=avg(np.linalg.norm(z['linear']-gt,axis=-1)*1408)
features={k:[] for k in ['context_fraction','endpoint_error_px','oracle_linear_error_px','roi_outside_fraction','roi_size_px','left_slot','right_slot']}
for ci,wi in enumerate(z['sample_indices']):
    gap=int(z['gaps'][ci]);lo=8-(gap-1)//2;hi=lo+gap;fids=index['feature_ids'][wi].numpy()
    obs=w['observed'][wi].numpy().copy();obs[lo:hi]=False
    left=np.where(obs&(np.arange(17)[:,None]<lo),np.arange(17)[:,None],-1).max(0)
    right=np.where(obs&(np.arange(17)[:,None]>=hi),np.arange(17)[:,None],17).min(0)
    both=(left>=0)&(right<17);m=mask[ci];lp=left.clip(0,16);rp=right.clip(0,16);js=np.arange(20)
    valid=both&vf[fids[lp],js]&vf[fids[rp],js]&m
    xy=w['xy'][wi].numpy();dt=w['dt'][wi].numpy()
    le=np.linalg.norm(xy[lp,js]-gtf[fids[lp],js],axis=-1)*1408
    rr=np.linalg.norm(xy[rp,js]-gtf[fids[rp],js],axis=-1)*1408
    weight=np.clip(-dt[lp]/np.maximum(dt[rp]-dt[lp],1e-6),0,1)
    oracle=gtf[fids[lp],js]*(1-weight[:,None])+gtf[fids[rp],js]*weight[:,None]
    oe=np.linalg.norm(oracle-gt[ci],axis=-1)*1408
    roi=z['roi'][ci];outside=((gt[ci]<roi[:2])|(gt[ci]>=roi[2:])).any(-1)
    vals=dict(context_fraction=float(both[m].mean()) if m.any() else 0,
        endpoint_error_px=float(((le+rr)/2)[valid].mean()) if valid.any() else np.nan,
        oracle_linear_error_px=float(oe[valid].mean()) if valid.any() else np.nan,
        roi_outside_fraction=float(outside[m].mean()) if m.any() else 0,roi_size_px=float((roi[2:]-roi[:2]).mean()*1408),
        left_slot=int(np.median(left[m&both])) if (m&both).any() else max(lo-1,0),
        right_slot=int(np.median(right[m&both])) if (m&both).any() else min(hi,16))
    for k,v in vals.items():features[k].append(v)
features={k:np.array(v) for k,v in features.items()};eligible=count>=3
recovered=eligible&(b>=30)&(e<=20)&(b-e>=10)
failed=eligible&(e>40)
harmed=eligible&(b<=20)&(e>40)
def stats(sel):
    n=int(sel.sum())
    if not n:return dict(cases=0)
    return dict(cases=n,dit_mean_px=float(e[sel].mean()),tracks_mean_px=float(b[sel].mean()),regression_mean_px=float(r[sel].mean()),
        recovery_rate_among_hard=float(recovered[sel].sum()/max(1,(sel&(b>=30)).sum())),
        under20_fraction=float((e[sel]<=20).mean()),over40_fraction=float((e[sel]>40).mean()),harmed_cases=int(harmed[sel].sum()),
        **{k:float(np.nanmedian(v[sel])) for k,v in features.items() if not k.endswith('slot')})
summary=dict(scope='Post-hoc mining of already evaluated test sequences, descriptive associations only; no retraining or new independent evidence',
    unit='case = one central hand window at one gap/mask; the same window repeats across settings',
    thresholds=dict(eligible='>=3 evaluated finger joints',recovered='tracks mean >=30px, RGB DiT mean <=20px and >=10px better',failed='RGB DiT mean >40px',harmed='tracks <=20px but RGB DiT >40px'),
    modes={},strata={},groups={})
for mode in ['partial','full','natural']:
    sel=eligible&(z['labels']==mode);summary['modes'][mode]=stats(sel)
    summary['strata'][mode]={}
    for name,cond in [('gap1',z['gaps']==1),('gap3',z['gaps']==3),('gap6',z['gaps']==6),('gap9',z['gaps']==9),
        ('endpoint_good_le20',features['endpoint_error_px']<=20),('endpoint_bad_gt40',features['endpoint_error_px']>40),
        ('motion_linear_le10',features['oracle_linear_error_px']<=10),('motion_nonlinear_gt30',features['oracle_linear_error_px']>30),
        ('both_sides_all',features['context_fraction']>=.99),('missing_side',features['context_fraction']<.99),
        ('small_roi_lt200',features['roi_size_px']<200),('large_roi_ge400',features['roi_size_px']>=400),
        ('whole_hand_visibility_lt_half',z['visibility']<.5),('whole_hand_visibility_ge_half',z['visibility']>=.5)]:
        summary['strata'][mode][name]=stats(sel&cond)
groups={f'{mode}_{kind}':eligible&(z['labels']==mode)&condition for mode in ['partial','full','natural'] for kind,condition in [('recovered',recovered),('failed',failed)]}
groups['harmed']=harmed
selected={}
def panel(ax,wi,slot,ci,occlude=False,overlay=False):
    fid=int(index['feature_ids'][wi,slot]);ax.axis('off')
    if not fid:ax.text(.1,.5,'No associated frame');return
    rec=records[fid-1];roi=index['roi'][fid].numpy()*1408;im=cv2.imread(rec['image'])
    if occlude:
        rect=index['rectangles'][fid,int(z['variant'][ci])].numpy();uv=roi[:2]+rect.reshape(2,2)*(roi[2:]-roi[:2])/256
        lo=np.maximum(np.floor(uv[0]).astype(int)-1,0);hi=np.minimum(np.ceil(uv[1]).astype(int)+1,[1408,1408])
        im[lo[1]:hi[1],lo[0]:hi[0]]=32+64*((rec['clip']+1)%4)
    ax.imshow(np.rot90(crop_image(im,roi)[:,:,::-1]));ax.set_xlim(0,255);ax.set_ylim(255,0)
    if overlay:
        for pts,color,marker in [(gt[ci],'#00ffca','o'),(z['predicted'][ci],'#ff5959','x')]:
            xy=(pts*1408-roi[:2])/(roi[2:]-roi[:2])*256;xy=np.stack([xy[:,1],255-xy[:,0]],-1)
            ax.scatter(xy[mask[ci],0],xy[mask[ci],1],s=16,c=color,marker=marker,linewidths=1)
for name,sel in groups.items():
    summary['groups'][name]=stats(sel);candidates=np.where(sel)[0];chosen=[];clips=set()
    # Representative central severity, with different clips; do not cherry-pick
    # only maximum improvement or maximum error examples.
    if len(candidates):
        target=np.median(e[candidates]);order=candidates[np.argsort(np.abs(e[candidates]-target))]
        for ci in order:
            wi=int(z['sample_indices'][ci]);meta=rows[wi];key=(meta['sequence'],meta['clip'])
            if key in clips:continue
            chosen.append(int(ci));clips.add(key)
            if len(chosen)==3:break
    selected[name]=[]
    if not chosen:continue
    fig,axs=plt.subplots(len(chosen),5,figsize=(14,3.2*len(chosen)),squeeze=False,layout='constrained')
    for row,ci in enumerate(chosen):
        wi=int(z['sample_indices'][ci]);l=int(features['left_slot'][ci]);rr=int(features['right_slot'][ci]);mode=str(z['labels'][ci])
        titles=[f'Past evidence ({l-8:+d} slots)','Current original: reference only' if mode!='natural' else 'Current RGB',
            'Current model-visible RGB','Reference green / DiT red',f'Future evidence ({rr-8:+d} slots)']
        for col,(slot,masked,overlay) in enumerate([(l,False,False),(8,False,False),(8,mode!='natural',False),(8,mode!='natural',True),(rr,False,False)]):
            panel(axs[row,col],wi,slot,ci,masked,overlay);axs[row,col].set_title(titles[col],fontsize=9)
        text=f"{rows[wi]['sequence']} / clip {rows[wi]['clip']} / frame {rows[wi]['frame']} | gap {z['gaps'][ci]} | DiT {e[ci]:.1f}, tracks {b[ci]:.1f}, regression {r[ci]:.1f} px"
        axs[row,2].set_xlabel(text,fontsize=9);axs[row,2].xaxis.set_visible(True);axs[row,2].set_axis_on();axs[row,2].set_xticks([]);axs[row,2].set_yticks([])
        selected[name].append(dict(case_index=ci,window_index=wi,sequence=rows[wi]['sequence'],clip=rows[wi]['clip'],frame=rows[wi]['frame'],gap=int(z['gaps'][ci]),
            dit_px=float(e[ci]),tracks_px=float(b[ci]),regression_px=float(r[ci]),**{k:float(v[ci]) for k,v in features.items()}))
    fig.suptitle(name+' | original hidden pixels shown for audit only',fontsize=12);fig.savefig(OUT/f'{name}.png',dpi=145);plt.close(fig)
s.save(OUT/'summary.json',summary);s.save(OUT/'selected_cases.json',selected)
np.savez_compressed(OUT/'case_metrics.npz',case_index=np.arange(len(e)),window_index=z['sample_indices'],mode=z['labels'],gap=z['gaps'],sequence=z['clusters'],count=count,
    dit_px=e,tracks_px=b,regression_px=r,linear_px=linear,recovered=recovered,failed=failed,harmed=harmed,**features)
print(json.dumps(summary,indent=2))
