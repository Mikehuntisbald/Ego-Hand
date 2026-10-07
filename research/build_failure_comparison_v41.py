"""Frozen v16 vs structured v39 failure visualizations; posthoc labels only.

No inference, training, threshold change, or default change. Original RGB
remains unmasked. Actual v39 fallbacks and rejected fit hypotheses differ.
"""
import collections, hashlib, html, json, shutil, zipfile
from pathlib import Path
import cv2, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hand3d_v8_common import V7, save
from hand3d_rollout_v8 import project_fisheye624
from joint_mano_model_v29 import EDGES
from evaluate_joint_kinematic_v30 import identities

ROOT=V7.parent; OUT=ROOT/'failure_comparison_v41'
JOINT_NAMES=['thumb tip','index tip','middle tip','ring tip','little tip','wrist',
             'thumb base','thumb IP','index MCP','index PIP','index DIP',
             'middle MCP','middle PIP','middle DIP','ring MCP','ring PIP','ring DIP',
             'little MCP','little PIP','little DIP']


def finite(values):
    a=np.asarray(values,dtype=float)
    return [finite(x) if np.ndim(x)>0 else (float(x) if np.isfinite(x) else None) for x in a]


def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);(OUT/'frames').mkdir(exist_ok=True)
    source=ROOT/'joint_mano_v28';allrows=json.loads((source/'rows.json').read_text())
    ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select']);rows=[allrows[i] for i in ids]
    n=len(rows);cache0=torch.load(source/'candidates.pt',weights_only=False,mmap=True)
    cache={k:v[ids] for k,v in cache0.items() if torch.is_tensor(v) and len(v)==len(allrows)}
    labels=torch.load(source/'evaluation_labels.pt',weights_only=False,mmap=True)
    gt,valid=labels['gt'][ids],labels['valid'][ids];mask=valid.clone();mask[:,5]=False
    result=torch.load(ROOT/'fitted_joint_refinement_v39/results.pt',weights_only=False)
    old,new,base=cache['baseline'],result['prediction'],cache['base'];accepted=result['frame_accepted'].bool()
    assert torch.equal(new[~accepted],old[~accepted]),'Every rejected frame must be a true v16 fallback'
    rejected=torch.load(ROOT/'fitted_parameter_trajectory_v39/whole_track.pt',weights_only=False)['hypothesis']
    side=identities(rows,dict(gt=gt,valid=valid))
    R,T=cache['rotation'],cache['translation']
    world=lambda x:torch.einsum('njc,nkc->njk',x,R)+T[:,None]
    pose=lambda x:world(x)-world(x)[:,5:6]
    variants={'gt':gt,'input':base,'old':old,'new':new,'rejected':rejected}
    poses={k:pose(x) for k,x in variants.items()}
    ce=lambda x:(x-gt).norm(dim=-1)*1000
    re=lambda x:((x-x[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
    cr={k:ce(x) for k,x in variants.items()};er={k:re(x) for k,x in variants.items()}
    mean=lambda e:(e*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    complete=valid.all(-1);om,nm=mean(er['old']),mean(er['new'])
    cmold,cmnew=mean(cr['old']),mean(cr['new'])
    ei,ej=zip(*EDGES);ev=valid[:,ei]&valid[:,ej]
    length={k:(x[:,ei]-x[:,ej]).norm(dim=-1)*1000 for k,x in variants.items()}
    ratio={k:v/length['gt'].clamp_min(1e-6) for k,v in length.items()}
    extreme={k:((v<.5)|(v>2))&ev for k,v in ratio.items()}
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    pairs=[]
    for indices in groups.values():
        indices.sort(key=lambda i:rows[i]['timestamp_ns'])
        for a,b in zip(indices,indices[1:]):
            if rows[b]['frame']-rows[a]['frame']==1 and side[a] is not None and side[a]==side[b]:pairs.append((a,b))
    a,b=map(torch.tensor,zip(*pairs));pm=valid[a]&valid[b];pm[:,5]=False
    moves={k:(x[b]-x[a]).norm(dim=-1)*1000 for k,x in poses.items()}
    jumps={k:pm&(moves['gt']<10)&(v>30) for k,v in moves.items()}
    bothaccepted=accepted[a]&accepted[b]
    recovery_loss=mask&(er['old']<=10)&(er['new']>20)
    camera_loss=mask&(cr['old']<=10)&(cr['new']>20)
    stats=dict(observations=n,complete_gt_frames=int(complete.sum()),valid_finger_points=int(mask.sum()),
        accepted_frames=int(accepted.sum()),fallback_frames=int((~accepted).sum()),same_hand_adjacent_pairs=len(pairs),
        v16_relative_good_points=int((mask&(er['old']<=10)).sum()),v16_relative_good_to_new_bad_points=int(recovery_loss.sum()),
        v16_relative_good_to_new_bad_frames=int(recovery_loss.any(-1).sum()),
        v16_camera_good_points=int((mask&(cr['old']<=10)).sum()),v16_camera_good_to_new_bad_points=int(camera_loss.sum()),
        frames_relative_mean_over20={'v16':int((complete&(om>20)).sum()),'v39_actual':int((complete&(nm>20)).sum())},
        bone_extreme_frames={'v16':int(extreme['old'].any(-1).sum()),'v39_actual':int(extreme['new'].any(-1).sum()),
                             'v39_accepted_only':int((extreme['new'].any(-1)&accepted).sum())},
        jump_pairs={'v16':int(jumps['old'].any(-1).sum()),'v39_actual':int(jumps['new'].any(-1).sum()),
                    'v39_accepted_only':int((jumps['new'].any(-1)&bothaccepted).sum())})
    # Separate selection criteria and measured counts. Categories overlap and
    # are descriptive examples, not runtime rules or visibility annotations.
    candidates=[]
    def frame_case(key,title,description,eligible,severity,focus='relative',edge=None):
        eligible=eligible&complete
        candidates.append(dict(key=key,title=title,description=description,count=int(eligible.sum()),unit='帧',
                               eligible=eligible,severity=severity,focus=focus,edge=edge,kind='frame'))
    frame_case('old_bone','旧版：骨架变形，新版规整',
               'v16存在相对GT的骨长极端值；新版接受的参数手型消除了该异常。消除骨架异常与定位更准要分别判断。',
               accepted&extreme['old'].any(-1)&~extreme['new'].any(-1),
               extreme['old'].sum(-1).float()*100+om,edge='worst_old')
    resolved=jumps['old']&~jumps['new']&bothaccepted[:,None]
    candidates.append(dict(key='old_jump',title='旧版：跳变被新版减轻',
        description='同一GT手的连续帧里，GT指端基本稳定而v16跳变；新版接受轨迹在这个点上低于跳变诊断阈值。',
        count=int(resolved.any(-1).sum()),unit='相邻帧对',eligible=resolved.any(-1),severity=(moves['old']*resolved).max(-1).values,
        pointmask=resolved,kind='pair'))
    persists=jumps['new']&bothaccepted[:,None]
    candidates.append(dict(key='new_jump',title='新版：正常手型仍会跳变',
        description='新版轨迹通过声明的结构/速度/加速度边界，但在GT基本稳定的点上仍发生>30mm跳变；有界运动不等于零跳变。',
        count=int(persists.any(-1).sum()),unit='相邻帧对',eligible=persists.any(-1),severity=(moves['new']*persists).max(-1).values,
        pointmask=persists,kind='pair'))
    frame_case('new_pose','新版：手型正常，但姿态更错',
               '新版已被接受、没有骨长极端，但手指相对误差比v16增加>5mm。骨架/指关节角可行，仍可能选错朝向或弯曲状态。',
               accepted&~extreme['new'].any(-1)&(nm-om>5),(nm-om),focus='regression')
    frame_case('lost_recovery','新版：丢失v16已恢复的点',
               '选中点在v16中相对误差≤10mm，在新版中变成>20mm。这里参照v16，和此前参照原WiLoR的准确点保护是不同统计。',
               accepted&recovery_loss.any(-1),((er['new']-er['old'])*recovery_loss).max(-1).values,focus='lost')
    frame_case('both_fail','两组：都未恢复正确手指',
               '两组腕相对平均误差均>30mm，且新版轨迹已接受。附近RGB帧只供观察，不把高误差自动当作不可见或整个片段无信息。',
               accepted&(om>30)&(nm>30),torch.minimum(om,nm))
    fallbackjump=torch.zeros(n,dtype=torch.bool);fallbackjump[b[jumps['new'].any(-1)]]=True
    frame_case('fallback','新版：拒绝候选后继承旧版失败',
               '新版未接受这条轨迹，实际输出逐字节保留v16。灰色虚线可显示未接受拟合假设，不能把假设当作最终结果或恢复成功。',
               ~accepted&(extreme['new'].any(-1)|fallbackjump|(nm>30)),nm+extreme['new'].sum(-1)*80,focus='relative')
    frame_case('root_error','两组：手型接近，整体位置偏',
               '新版相对误差<15mm但相机误差>40mm。腕相对手型可能接近，整手位置/深度仍错；下方同时报告相机误差和腕点深度。',
               accepted&(nm<15)&(cmnew>40),cmnew,focus='root')
    used=[];cases=[]
    colors=dict(gt='#26915b',input='#78828e',old='#ed8b23',new='#356fd1',rejected='#8f66af')
    for item in candidates:
        possible=torch.nonzero(item['eligible']).flatten().tolist()
        if not possible:continue
        possible.sort(key=lambda i:float(item['severity'][i]),reverse=True)
        def is_distinct(idx):
            ri=int(b[idx]) if item['kind']=='pair' else idx;r=rows[ri]
            return all((r['sequence'],r['clip'],r['track_id'])!=(rows[j]['sequence'],rows[j]['clip'],rows[j]['track_id']) for j in used)
        def context_rich(idx):
            ri=int(b[idx]) if item['kind']=='pair' else idx;r=rows[ri]
            return len(groups[(r['sequence'],r['clip'],r['track_id'])])>=12
        selected=next((i for i in possible if is_distinct(i) and context_rich(i)),
                      next((i for i in possible if context_rich(i)),possible[0]))
        if item['kind']=='pair':
            focus_index=int(b[selected]);first_index=int(a[selected]);m=item['pointmask'][selected]
            method='old' if item['key']=='old_jump' else 'new'
            joint=int((moves[method][selected]*m).argmax());event_frames=[rows[first_index]['frame'],rows[focus_index]['frame']]
        else:
            focus_index=selected;first_index=selected;event_frames=[rows[selected]['frame']]
            if item['focus']=='lost':joint=int(((er['new']-er['old'])*recovery_loss)[selected].argmax())
            elif item['focus']=='regression':joint=int((er['new']-er['old'])[selected].masked_fill(~mask[selected],-1e9).argmax())
            elif item['focus']=='root':joint=5
            else:joint=int(torch.maximum(er['old'],er['new'])[selected].masked_fill(~mask[selected],-1).argmax())
        used.append(focus_index);r=rows[focus_index];key=(r['sequence'],r['clip'],r['track_id']);g=groups[key]
        location=g.index(focus_index);clipids=g[max(0,location-15):min(len(g),location+16)]
        edge=next((e for e,(u,v) in enumerate(EDGES) if v==joint or u==joint),0)
        if item.get('edge')=='worst_old':
            severity=torch.maximum(ratio['old'][focus_index],1/ratio['old'][focus_index].clamp_min(1e-6))*extreme['old'][focus_index]
            edge=int(severity.argmax());joint=int(ej[edge])
        boxes=np.array([rows[i]['box'] for i in clipids]);lo=boxes[:,:2].min(0);hi=boxes[:,2:].max(0)
        crop_size=int(min(1408,max(hi-lo)*1.28));crop_size=max(200,crop_size)
        crop_xy=np.clip(np.floor((lo+hi)/2-crop_size/2).astype(int),0,1408-crop_size)
        scale=512/crop_size;serial=f'{len(cases)+1:02d}'
        frames=[];center_file=f'full_{serial}.jpg';shutil.copyfile(r['image'],OUT/center_file)
        for local,i in enumerate(clipids):
            image=cv2.imread(rows[i]['image']);assert image is not None
            # Follow each predicted hand box for a readable finger closeup.
            # Motion diagnostics use worldXYZ, never these changing crops.
            box=np.asarray(rows[i]['box'],float)
            crop_size=int(min(1408,max(200,max(box[2:]-box[:2])*1.7)))
            crop_xy=np.clip(np.floor((box[:2]+box[2:])/2-crop_size/2).astype(int),0,1408-crop_size)
            scale=512/crop_size
            x0,y0=map(int,crop_xy);image=image[y0:y0+crop_size,x0:x0+crop_size]
            image=cv2.resize(image,(512,512),interpolation=cv2.INTER_AREA)
            filename=f'frames/c{serial}_{local:03d}.jpg';assert cv2.imwrite(str(OUT/filename),image,[cv2.IMWRITE_JPEG_QUALITY,91])
            frame=dict(frame=rows[i]['frame'],time_s=(rows[i]['timestamp_ns']-r['timestamp_ns'])/1e9,image=filename,
                accepted=bool(accepted[i]),complete_gt=bool(complete[i]),box_confidence=float(rows[i]['score']),
                metrics={},points={},world_relative={},root_camera={})
            for name,x in variants.items():
                uv=project_fisheye624(x[i:i+1],cache['camera_params'][i:i+1])[0].numpy()
                uv=(uv-crop_xy)*scale;relpose=poses[name][i].numpy()*1000
                if name=='gt':uv[~valid[i].numpy()]=np.nan;relpose[~valid[i].numpy()]=np.nan
                frame['points'][name]=finite(uv);frame['world_relative'][name]=finite(relpose)
                frame['root_camera'][name]=finite(x[i,5].numpy()*1000) if name!='gt' or valid[i,5] else [None]*3
                frame['metrics'][name]=dict(camera_mm=float(mean(cr[name])[i]) if complete[i] else None,
                    relative_mm=float(mean(er[name])[i]) if complete[i] else None,
                    focus_camera_mm=float(cr[name][i,joint]) if valid[i,joint] else None,
                    focus_relative_mm=float(er[name][i,joint]) if valid[i,joint] else None,
                    bone_mm=float(length[name][i,edge]),bone_gt_ratio=float(ratio[name][i,edge]) if ev[i,edge] else None)
            frames.append(frame)
        focus_local=clipids.index(focus_index)
        cloud=np.concatenate([poses[k][clipids].numpy().reshape(-1,3)*1000 for k in ['gt','input','old','new']])
        midpoint=(cloud.min(0)+cloud.max(0))/2;rad=float(np.linalg.norm(cloud-midpoint,axis=-1).max())*1.08
        case=dict(id=item['key'],title=item['title'],description=item['description'],eligible_count=item['count'],eligible_unit=item['unit'],
            sequence=r['sequence'],clip=r['clip'],track_id=r['track_id'],focus_joint=joint,focus_joint_name=JOINT_NAMES[joint],
            bone_edge=list(EDGES[edge]),focus=focus_local,event_frames=event_frames,accepted=bool(accepted[focus_index]),
            full_image=center_file,frames=frames,radius_mm=max(60,rad),center_mm=midpoint.tolist())
        cases.append(case)
        snapshot(OUT,case,variants,poses,cr,er,mask,cache,focus_index,crop_xy,crop_size,colors)
        curveposes={k:world(x) for k,x in variants.items()} if joint==5 else poses
        curves(OUT,case,clipids,rows,curveposes,length,cr if joint==5 else er,mask,joint,edge,colors)
        print(json.dumps(dict(case=item['key'],clip=r['clip'],frame=r['frame'],accepted=case['accepted'],count=item['count']),ensure_ascii=False),flush=True)
    summary=dict(complete=True,old='v16 final XYZ refinement',new='v39 fitted parameter regression plus wholetrack feasible refinement; actualoutput includes fallbacks',
        scope='Posthoc characterization on existing dev_select only, 4sequences ofP0003. No independent validation, no model/default changes.',stats=stats,
        categories=[{k:v for k,v in x.items() if k in ['key','title','description','count','unit']} for x in candidates],
        methodology={'relative_failure_mean_threshold_mm':20,'case_both_failure_relative_mean_mm':30,'case_pose_worsening_min_mm':5,
                     'good_to_bad':'v16 point error<=10mm -> v39 actual point error>20mm; reference differs from original WiLoR preservation',
                     'jump':'sameGThand consecutiveframe, GT relativeworldmovement<10mm and predicted>30mm',
                     'bone_extreme':'predicted length<0.5x or>2x GT','selection':'Largest category severity; prefer>=12-observation track and another track for richer context. Diagnostic examples, not unbiased samples. Categories overlap.',
                     'rgb':'Unmasked RGB closeups follow each predictedbox(1.7x); worldXYZ motion never uses crop displacement. Fulloriginal center images copied intact. No artificial occlusion.',
                     'new_candidate':'Gray dashed rejectedhypothesis is not actual adoptedoutput. Accepted flag concerns feasible trajectory only, not final model adoption.'},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'rows.json',source/'candidates.pt',source/'evaluation_labels.pt',ROOT/'fitted_joint_refinement_v39/results.pt']},
        cases=[{k:v for k,v in c.items() if k!='frames'} for c in cases])
    save(OUT/'summary.json',summary)
    payload=dict(summary=summary,cases=cases,edges=[list(e) for e in EDGES],joint_names=JOINT_NAMES)
    encoded=json.dumps(payload,separators=(',',':'),ensure_ascii=False,allow_nan=False).replace('</',r'<\/')
    template=(Path(__file__).parent/'failure_compare_v41.html').read_text(encoding='utf-8')
    assert '__REPORT_DATA__' in template
    (OUT/'report.html').write_text(template.replace('__REPORT_DATA__',encoded),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'failure_comparison_v41.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(OUT)))
    print(json.dumps(dict(complete=True,cases=len(cases),stats=stats,archive=str(ROOT/'failure_comparison_v41.zip')),ensure_ascii=False),flush=True)


def snapshot(out,case,variants,poses,cr,er,mask,cache,n,xy,size,colors):
    image=cv2.imread(str(out/case['frames'][case['focus']]['image']))
    fig=plt.figure(figsize=(14,7));names=['input','gt','old','new'];titles=['OriginalRGB / WiLoR input','GT','v16','v39 accepted' if case['accepted'] else 'v39 FALLBACK = v16']
    topnames=[None,'gt','old','new'];joint=case['focus_joint']
    for col,name in enumerate(topnames):
        ax=fig.add_subplot(2,4,col+1);ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        if name:
            pts=case['frames'][case['focus']]['points'][name];u=np.array([[np.nan,np.nan] if p[0] is None else p for p in pts])
            for a,b in EDGES:ax.plot(u[[a,b],0],u[[a,b],1],color=colors[name],lw=1.6)
            ax.scatter(u[:,0],u[:,1],color=colors[name],s=9)
            ax.scatter(u[joint,0],u[joint,1],facecolors='none',edgecolors='#f0dc2e',linewidths=2,s=90)
        ax.set_xlim(0,512);ax.set_ylim(512,0);ax.axis('off');ax.set_title(titles[col],fontsize=10)
        k=names[col];pose=poses[k][n].numpy()*1000;ax=fig.add_subplot(2,4,col+5,projection='3d')
        for a,b in EDGES:ax.plot(pose[[a,b],0],pose[[a,b],1],pose[[a,b],2],c=colors[k],lw=2)
        ax.scatter(*pose.T,c=colors[k],s=9);ax.scatter(*pose[joint],c='#d7b400',s=45)
        rad=case['radius_mm'];mid=case['center_mm'];ax.set_xlim(mid[0]-rad,mid[0]+rad);ax.set_ylim(mid[1]-rad,mid[1]+rad);ax.set_zlim(mid[2]-rad,mid[2]+rad)
        ax.set_box_aspect([1,1,1]);ax.view_init(elev=22,azim=-65);ax.set_xlabel('WorldX(mm)');ax.set_ylabel('WorldY(mm)');ax.set_zlabel('WorldZ(mm)')
        ax.set_title(f'{k}: camera {float((cr[k][n]*mask[n]).sum()/mask[n].sum()):.1f} / relative {float((er[k][n]*mask[n]).sum()/mask[n].sum()):.1f}mm\nSelectedpoint relativeerror {float(er[k][n,joint]):.1f}mm',fontsize=9)
    fig.suptitle(f'clip{case["clip"]} track{case["track_id"]} frame{case["event_frames"][-1]} | {case["id"]} | focus={JOINT_NAMES[joint]}',fontsize=12)
    fig.tight_layout();name=f'snapshot_{case["id"]}.png';fig.savefig(out/name,dpi=145,bbox_inches='tight');plt.close(fig);case['snapshot']=name


def curves(out,case,ids,rows,poses,length,er,mask,joint,edge,colors):
    t=np.array([(rows[i]['timestamp_ns']-rows[ids[case['focus']]]['timestamp_ns'])/1e9 for i in ids]);fig,axes=plt.subplots(1,3,figsize=(14,3.5))
    gtvalid=np.array([case['frames'][j]['complete_gt'] for j in range(len(ids))])
    for k in ['gt','old','new']:
        z=poses[k][ids,joint].numpy();move=np.linalg.norm(np.diff(z,axis=0),axis=-1)*1000
        okay=gtvalid[1:]&gtvalid[:-1]
        move[~okay]=np.nan;axes[0].plot(t[1:],move,color=colors[k],label=k,lw=1.8)
        y=length[k][ids,edge].numpy();y[~gtvalid]=np.nan;axes[1].plot(t,y,color=colors[k],label=k,lw=1.8)
        if k!='gt':
            y=(er[k][ids]*mask[ids]).sum(-1)/mask[ids].sum(-1).clamp_min(1);y=y.numpy();y[~gtvalid]=np.nan
            axes[2].plot(t,y,color=colors[k],label=k,lw=1.8)
    axes[0].axhline(30,color='#a5abb4',linestyle='--',lw=1);axes[0].set_title(f'{JOINT_NAMES[joint]} world{"" if joint==5 else "-relative"} step(mm)')
    axes[1].set_title(f'Bone {list(EDGES[edge])} length(mm)');axes[2].set_title('Mean camera error(mm)' if joint==5 else 'Mean relativefinger error(mm)')
    for ax in axes:
        ax.axvline(0,color='#4a5361',linestyle=':',lw=1);ax.set_xlabel('Time from focusframe(s)');ax.set_ylim(bottom=0);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();name=f'curves_{case["id"]}.png';fig.savefig(out/name,dpi=145,bbox_inches='tight');plt.close(fig);case['curves']=name


if __name__=='__main__':main()
