"""Unmasked real RGB, shape/motion comparisons, and explicit accuracy costs."""
import argparse,collections,json,shutil,zipfile
from pathlib import Path
import cv2,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hand3d_v8_common import V7,save
from hand3d_rollout_v8 import project_fisheye624
from joint_mano_model_v29 import EDGES
from evaluate_joint_kinematic_v30 import identities

COLORS=dict(gt='#228c60',v16='#dd8727',v39='#765cbe',v42='#216bd8')
TITLES=dict(gt='GT',v16='v16',v39='v39 actual',v42='v42 stable')


def clean(value):
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(x) for x in value]
    return float(value) if np.isfinite(value) else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='stability_first_v42_sidefix');args=ap.parse_args()
    torch.set_num_threads(4);root=V7.parent;run=root/args.run;out=root/'stability_review_v42'
    out.mkdir(exist_ok=True);(out/'frames').mkdir(exist_ok=True)
    allrows=json.loads((root/'joint_mano_v28/rows.json').read_text());ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select']);rows=[allrows[i] for i in ids]
    ca=torch.load(root/'joint_mano_v28/candidates.pt',weights_only=False,mmap=True)
    cache={k:v[ids] for k,v in ca.items() if torch.is_tensor(v) and len(v)==len(allrows)}
    la=torch.load(root/'joint_mano_v28/evaluation_labels.pt',weights_only=False,mmap=True);gt=la['gt'][ids];valid=la['valid'][ids]
    frozen=torch.load(run/'results.pt',weights_only=False);summary=json.loads((run/'summary.json').read_text());summary['verification']=json.loads((run/'verification.json').read_text())
    variants=dict(gt=gt,v16=cache['baseline'],v39=torch.load(root/'fitted_joint_refinement_v39/results.pt',weights_only=False)['prediction'],v42=frozen['prediction'])
    world=lambda x:torch.einsum('njc,nkc->njk',x,cache['rotation'])+cache['translation'][:,None]
    worlds={k:world(x) for k,x in variants.items()};poses={k:x-x[:,5:6] for k,x in worlds.items()}
    errors={k:((x-x[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000 for k,x in variants.items()}
    mask=valid.clone();mask[:,5]=False;complete=valid.all(-1)
    means={k:(v*mask).sum(-1)/mask.sum(-1).clamp_min(1) for k,v in errors.items()}
    side=identities(rows,dict(gt=gt,valid=valid));groups=collections.defaultdict(list);pairs=[]
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    for ix in groups.values():
        ix.sort(key=lambda i:rows[i]['timestamp_ns'])
        for a,b in zip(ix,ix[1:]):
            if rows[b]['frame']-rows[a]['frame']==1 and side[a] is not None and side[a]==side[b]:pairs.append((a,b))
    a,b=map(torch.tensor,zip(*pairs));pm=valid[a]&valid[b];pm[:,5]=False
    moves={k:(x[b]-x[a]).norm(dim=-1)*1000 for k,x in poses.items()}
    jump39=pm&(moves['gt']<10)&(moves['v39']>30)
    retention=moves['v42']/moves['gt'].clamp_min(1e-8)
    fast=pm&(moves['gt']>10);lost=mask&(errors['v16']<=10)&(errors['v42']>20)
    ei,ej=zip(*EDGES);gl=(gt[:,ei]-gt[:,ej]).norm(dim=-1);vl=(variants['v16'][:,ei]-variants['v16'][:,ej]).norm(dim=-1)
    extreme=(((vl/gl.clamp_min(1e-7))<.5)|((vl/gl.clamp_min(1e-7))>2))&valid[:,ei]&valid[:,ej]
    definitions=[]
    def pair_case(key,title,text,condition,severity,joint=0):
        hits=torch.nonzero(condition.any(-1)).flatten().tolist()
        hits.sort(key=lambda j:float(severity[j].max()),reverse=True)
        definitions.append(dict(key=key,title=title,text=text,count=len(hits),unit='相邻帧对',hits=hits,kind='pair',severity=severity,condition=condition,joint=joint))
    def frame_case(key,title,text,condition,severity):
        hits=torch.nonzero(condition&complete).flatten().tolist();hits.sort(key=lambda j:float(severity[j]),reverse=True)
        definitions.append(dict(key=key,title=title,text=text,count=len(hits),unit='帧',hits=hits,kind='frame'))
    pair_case('jump_resolved','v39 跳变 → v42 连续','同一GT手的连续帧；GT相对移动<10mm、v39>30mm。这里检查实际输出，未剔除旧回退。',jump39,moves['v39']*jump39)
    root39=(worlds['v39'][b,5]-worlds['v39'][a,5]).norm(dim=-1)*1000
    rootgt=(worlds['gt'][b,5]-worlds['gt'][a,5]).norm(dim=-1)*1000
    rootbad=((root39>80)&(rootgt<20)&valid[a,5]&valid[b,5])[:,None]
    pair_case('root_resolved','整手位置尖峰 → 受限运动','手腕/整手移动单独约束，不能只检查腕相对手指。减少尖峰不表示深度准确。',rootbad,root39[:,None]*rootbad,joint=5)
    frame_case('bone_resolved','旧骨架变形 → 参数手型','v16骨长相对GT<0.5或>2倍，展示固定手型与有界关节角生成的v42。不是碰撞或完全解剖学证明。',extreme.any(-1),extreme.sum(-1).float()*100+means['v16'])
    compressed=fast&(retention<.5)
    pair_case('fast_attenuated','代价：真实快速动作被压低','GT指端移动>10mm，而v42保留不足一半幅度。这些点不是稳定成功的例子；平滑会损失运动细节。',compressed,(moves['gt']-moves['v42'])*compressed)
    frame_case('accuracy_loss','代价：稳定但点位变差','v16相对误差≤10mm的点被v42改到>20mm，清楚展示稳定与准确之间的代价。',lost.any(-1),((errors['v42']-errors['v16'])*lost).max(-1).values)
    frame_case('still_wrong','仍失败：连续但姿态不对','v39和v42相对平均误差都>30mm。RGB仅供观察，不把高误差当成真实逐指不可见标签。',(means['v39']>30)&(means['v42']>30),means['v42'])
    retained=fast&(retention>.8)&(retention<1.2)
    pair_case('fast_retained','真实动作基本保留','GT移动>10mm且v42幅度保留80–120%；这里只表示幅度相近，方向和关节点位要另看。',retained,moves['gt']*retained)
    cases=[];used=set()
    for item in definitions:
        if not item['hits']:continue
        selected=None
        for candidate in item['hits']:
            n=int(b[candidate]) if item['kind']=='pair' else candidate;r=rows[n];key=(r['sequence'],r['clip'],r['track_id'])
            if key not in used and len(groups[key])>=12:selected=candidate;break
        if selected is None:selected=item['hits'][0]
        focus=int(b[selected]) if item['kind']=='pair' else selected;r=rows[focus];key=(r['sequence'],r['clip'],r['track_id']);used.add(key)
        g=groups[key];location=g.index(focus);ix=g[max(0,location-15):min(len(g),location+16)]
        if item['kind']=='pair':joint=item['joint'] if item['joint']==5 else int(item['severity'][selected].argmax())
        elif item['key']=='accuracy_loss':joint=int(((errors['v42']-errors['v16'])*lost)[focus].argmax())
        else:joint=int(errors['v42'][focus].masked_fill(~mask[focus],-1).argmax())
        serial=len(cases)+1;full=f'full_{serial:02d}.jpg';shutil.copyfile(r['image'],out/full);frames=[]
        for fi,n in enumerate(ix):
            raw=cv2.imread(rows[n]['image']);box=np.asarray(rows[n]['box']);size=int(min(1408,max(200,max(box[2:]-box[:2])*1.7)))
            xy=np.clip(np.floor((box[:2]+box[2:])/2-size/2).astype(int),0,1408-size);x,y=xy
            crop=cv2.resize(raw[y:y+size,x:x+size],(512,512),interpolation=cv2.INTER_AREA)
            filename=f'frames/c{serial:02d}_{fi:03d}.jpg';assert cv2.imwrite(str(out/filename),crop,[cv2.IMWRITE_JPEG_QUALITY,90])
            f=dict(frame=rows[n]['frame'],time_s=(rows[n]['timestamp_ns']-r['timestamp_ns'])/1e9,image=filename,points={},pose={},relative_mm={},complete_gt=bool(complete[n]))
            for name,p in variants.items():
                uv=project_fisheye624(p[n:n+1],cache['camera_params'][n:n+1])[0].numpy();uv=(uv-xy)*512/size
                pose=poses[name][n].numpy()*1000
                if name=='gt':uv[~valid[n].numpy()]=np.nan;pose[~valid[n].numpy()]=np.nan
                f['points'][name]=clean(uv);f['pose'][name]=clean(pose);f['relative_mm'][name]=float(means[name][n]) if complete[n] else None
            frames.append(f)
        cloud=np.concatenate([poses[k][ix].numpy().reshape(-1,3)*1000 for k in variants]);cloud=cloud[np.isfinite(cloud).all(-1)]
        center=(cloud.min(0)+cloud.max(0))/2;radius=float(np.linalg.norm(cloud-center,axis=-1).max())*1.05
        case=dict(id=item['key'],title=item['title'],description=item['text'],eligible_count=item['count'],eligible_unit=item['unit'],
                  sequence=r['sequence'],clip=r['clip'],track_id=r['track_id'],focus_joint=joint,focus=ix.index(focus),frames=frames,
                  full_image=full,center_mm=center.tolist(),radius_mm=max(60,radius))
        fig=plt.figure(figsize=(15,7.3));focusframe=frames[case['focus']];im=cv2.cvtColor(cv2.imread(str(out/focusframe['image'])),cv2.COLOR_BGR2RGB)
        for col,name in enumerate(['gt','v16','v39','v42']):
            ax=fig.add_subplot(2,4,col+1);ax.imshow(im);p=np.asarray([[np.nan]*2 if z[0] is None else z for z in focusframe['points'][name]])
            for u,v in EDGES:ax.plot(p[[u,v],0],p[[u,v],1],color=COLORS[name],lw=1.6)
            ax.scatter(p[:,0],p[:,1],s=9,c=COLORS[name]);ax.scatter(*p[joint],s=95,facecolors='none',edgecolors='#dbc727',lw=2)
            ax.set_xlim(0,512);ax.set_ylim(512,0);ax.axis('off');ax.set_title(TITLES[name])
            ax=fig.add_subplot(2,4,col+5,projection='3d');z=poses[name][focus].numpy()*1000
            if name=='gt':z[~valid[focus].numpy()]=np.nan
            for u,v in EDGES:ax.plot(*z[[u,v]].T,color=COLORS[name],lw=2)
            ax.scatter(*z.T,s=9,c=COLORS[name]);rad=case['radius_mm'];mid=case['center_mm']
            ax.set_xlim(mid[0]-rad,mid[0]+rad);ax.set_ylim(mid[1]-rad,mid[1]+rad);ax.set_zlim(mid[2]-rad,mid[2]+rad);ax.set_box_aspect([1,1,1]);ax.view_init(22,-65)
            ax.set_xlabel('X mm',fontsize=8,labelpad=-3);ax.set_ylabel('Y mm',fontsize=8,labelpad=-3);ax.set_zlabel('Z mm',fontsize=8,labelpad=-3);ax.tick_params(labelsize=8,pad=1);ax.set_title(f'Relative error: {means[name][focus]:.1f} mm',fontsize=10)
        fig.suptitle(f'{item["key"]} | clip{r["clip"]} track{r["track_id"]} frame{r["frame"]} focus joint{joint}');fig.tight_layout()
        snap=f'snapshot_{item["key"]}.png';fig.savefig(out/snap,dpi=135,bbox_inches='tight',pad_inches=.35);plt.close(fig);case['snapshot']=snap
        t=np.array([f['time_s'] for f in frames]);fig,axes=plt.subplots(1,3,figsize=(14,3.4))
        edge=next((e for e,(u,v) in enumerate(EDGES) if joint==v or joint==u),0);u,v=EDGES[edge]
        for name in variants:
            point=(worlds[name][ix,5] if joint==5 else poses[name][ix,joint]).clone()
            if name=='gt':point[~valid[ix,joint]]=float('nan')
            step=(point[1:]-point[:-1]).norm(dim=-1)*1000
            axes[0].plot(t[1:],step.numpy(),label=name,c=COLORS[name]);length=(variants[name][ix,u]-variants[name][ix,v]).norm(dim=-1)*1000
            if name=='gt':length[~(valid[ix,u]&valid[ix,v])]=float('nan')
            axes[1].plot(t,length.numpy(),label=name,c=COLORS[name]);err=means[name][ix].clone();err[~complete[ix]]=float('nan');axes[2].plot(t,err.numpy(),label=name,c=COLORS[name])
        for ax,title in zip(axes,['Root world step mm' if joint==5 else 'Finger wrist-relative world step mm',f'Bone{u}-{v} length mm','Relative finger error mm']):
            ax.set_title(title);ax.set_xlabel('Time s');ax.set_ylim(bottom=0);ax.axvline(0,color='#999',ls=':');ax.grid(alpha=.25);ax.legend()
        fig.tight_layout();curve=f'curves_{item["key"]}.png';fig.savefig(out/curve,dpi=140,bbox_inches='tight');plt.close(fig);case['curves']=curve
        cases.append(case)
    payload=dict(summary=summary,cases=cases,edges=[list(e) for e in EDGES],colors=COLORS)
    text=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</',r'<\/')
    template=(Path(__file__).parent/'stability_compare_v42.html').read_text();(out/'report.html').write_text(template.replace('__REPORT_DATA__',text),encoding='utf-8')
    for name in ['summary.json','verification.json','constraint_check.json','protocol.json','restoration.json']:
        if (run/name).exists():shutil.copyfile(run/name,out/name)
    save(out/'case_manifest.json',[{k:v for k,v in c.items() if k!='frames'} for c in cases])
    with zipfile.ZipFile(root/'stability_review_v42.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(out)))
    print(json.dumps(dict(cases=len(cases),frames=sum(len(c['frames']) for c in cases),archive=str(root/'stability_review_v42.zip'))),flush=True)


if __name__=='__main__':main()
